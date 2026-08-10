"""
The PearlHash mining search: noisy GEMM -> per-tile jackpot transcript ->
keyed BLAKE3 PoW check.

Mirrors zk-pow `try_mine_one` / the GPU `headless_mine_kernel`:

  * A (m x k), B (k x n) are random int8 signal matrices in [-64, 63]
  * noise E_A/E_B (low-rank) is added -> A_noised/B_noised (int16)
  * C is accumulated in chunks of `rank` along k; after every chunk the
    XOR-reduction of each pattern tile (rows_pattern.size() x
    cols_pattern.size() = 2 x 64) is rotate-XOR-ed into a 16-word
    "jackpot" transcript (rotate by 13)
  * jackpot_hash = blake3(transcript64, key=a_noise_seed)
  * if little-endian jackpot_hash <= target  -> a share is found

Matmul runs on torch (GPU) when available, numpy otherwise. The small
tile-XOR/rotate work happens on the CPU (numpy, uint32) which is fine for
the demo scale.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .commitment import commitment_seeds, hash_le32, jackpot_hash
from .noise import NoiseGenerator
from .params import (
    DEFAULT_COLS_PATTERN,
    DEFAULT_ROWS_PATTERN,
    IncompleteBlockHeader,
    MiningConfiguration,
    SIGNAL_MAX,
    SIGNAL_MIN,
)


def _backend_available(name: str) -> bool:
    if name == "torch":
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False
    return True


def _make_matmul(backend: str):
    """Return (dot, acc, xor_rows, to_cpu) helpers for the backend.

    CUDA has NO int32 GEMM kernel ("addmm_cuda not implemented for 'Int'"),
    so on torch we compute each rank-sized chunk in float32 and convert back
    to int32. This is bit-exact: inputs are bounded by ±127, so a rank=128
    chunk dot-product is at most 128*127*127 ~= 2.1e6 < 2^24, meaning every
    partial sum is exactly representable in float32. Accumulation across
    chunks happens in int32 (elementwise add IS supported on CUDA).
    """
    if backend == "torch":
        import torch

        def dot(A, B):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.from_numpy(np.ascontiguousarray(A)).to(dev).to(torch.float32)
            b = torch.from_numpy(np.ascontiguousarray(B)).to(dev).to(torch.float32)
            return torch.matmul(a, b).to(torch.int32)

        def acc(acc, chunk):
            return acc + chunk

        def xor_rows(C, o0, o1):
            """XOR pairs of rows C[o0[r], :] ^ C[o1[r], :] -> numpy (R, n) uint32."""
            r0 = C[o0.tolist()]
            r1 = C[o1.tolist()]
            x = torch.bitwise_xor(r0, r1)
            x64 = x.to(torch.int64).cpu().numpy()
            return (x64 & 0xFFFFFFFF).astype(np.uint32)

        def to_cpu(C):
            return C.cpu().numpy()

        return dot, acc, xor_rows, to_cpu

    def dot(A, B):
        return A.astype(np.int32) @ B.astype(np.int32)

    def acc(acc_, chunk):
        return acc_ + chunk

    def xor_rows(C, o0, o1):
        x = np.bitwise_xor(C[o0], C[o1])
        return x.view(np.uint32)

    def to_cpu(C):
        return C

    return dot, acc, xor_rows, to_cpu


def rotl32(x: np.ndarray, n: int) -> np.ndarray:
    return ((x << n) | (x >> (32 - n))) & np.uint32(0xFFFFFFFF)


class JackpotSearcher:
    """Computes per-tile jackpot transcripts for one noisy matmul."""

    def __init__(self, config: MiningConfiguration, m: int, n: int):
        self.config = config
        self.m = m
        self.n = n
        self.rank = config.rank
        rows_base = config.rows_pattern.to_list()
        cols_base = config.cols_pattern.to_list()
        self.rows_base = rows_base
        self.cols_base = cols_base

        rmax = config.rows_pattern.max()
        cmax = config.cols_pattern.max()
        self.rows_offs = np.array(
            [o for o in range(0, m - rmax) if config.rows_pattern.offset_is_valid(o)],
            dtype=np.int64,
        )
        self.cols_offs = np.array(
            [o for o in range(0, n - cmax) if config.cols_pattern.offset_is_valid(o)],
            dtype=np.int64,
        )
        self.R = len(self.rows_offs)
        self.C = len(self.cols_offs)
        self.num_chunks = config.common_dim // self.rank
        assert config.common_dim % self.rank == 0

        # o0/o1: the two opened rows per tile
        self.o0 = self.rows_offs + rows_base[0]
        self.o1 = self.rows_offs + rows_base[-1]
        # column index table for the gather
        self.col_idx = (self.cols_offs[:, None] + np.array(cols_base)[None, :]).astype(np.int64)

    # ------------------------------------------------------------------ #
    def search(self, A_noised: np.ndarray, B_noised: np.ndarray,
               backend: str = "auto") -> np.ndarray:
        """Return jackpot transcript tensor (R, C, 16) uint32."""
        if backend == "auto":
            backend = "torch" if _backend_available("torch") else "numpy"
        if backend not in ("torch", "numpy"):
            raise ValueError(f"unknown backend {backend}")

        dot, acc, xor_rows, _ = _make_matmul(backend)

        m, k = A_noised.shape
        n = B_noised.shape[1]
        C_acc = np.zeros((m, n), dtype=np.int32)
        if backend == "torch":
            import torch
            C_acc = torch.zeros((m, n), dtype=torch.int32,
                                device="cuda" if torch.cuda.is_available() else "cpu")

        jackpot = np.zeros((self.R, self.C, 16), dtype=np.uint32)

        for p in range(self.num_chunks):
            ks = slice(p * self.rank, (p + 1) * self.rank)
            C_chunk = dot(A_noised[:, ks], B_noised[ks, :])
            C_acc = acc(C_acc, C_chunk)

            xr = xor_rows(C_acc, self.o0, self.o1)          # (R, n) uint32
            xc = xr[:, self.col_idx]                        # (R, C, 64)
            xored = np.bitwise_xor.reduce(xc, axis=2)       # (R, C)

            tid = p % 16
            jackpot[:, :, tid] = rotl32(jackpot[:, :, tid], 13) ^ xored

        return jackpot

    # ------------------------------------------------------------------ #
    def tile_positions(self):
        """Enumerate (row_off, col_off) tiles in a deterministic order."""
        for ri, ro in enumerate(self.rows_offs):
            for ci, co in enumerate(self.cols_offs):
                yield ri, ci, int(ro), int(co)


def _default_rng(header: IncompleteBlockHeader, nonce: int):
    import blake3 as _b3
    seed = _b3.blake3(header.to_bytes() + int(nonce).to_bytes(8, "little")).digest()[:16]
    return np.random.default_rng(int.from_bytes(seed, "big"))


def mine_once(
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    m: int,
    n: int,
    target: int,
    nonce: int,
    backend: str = "auto",
    max_hashes: int | None = None,
    on_attempt=None,
):
    """One full mining attempt (fresh A/B) -> share dict or None."""
    k = config.common_dim
    rng = _default_rng(header, nonce)

    A = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(m, k), dtype=np.int8)
    B = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1, size=(k, n), dtype=np.int8)

    b_seed, a_seed = commitment_seeds(header, config, A, B)

    gen = NoiseGenerator(noise_rank=config.rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate_noise_matrices(
        key_A=a_seed, key_B=b_seed,
        A_rows=m, common_dim=k, B_cols=n,
    )
    E_A = (E_AL.astype(np.int32) @ E_AR.astype(np.int32)).astype(np.int8)
    E_B = (E_BL.astype(np.int32) @ E_BR.astype(np.int32)).astype(np.int8)
    A_noised = A.astype(np.int16) + E_A.astype(np.int16)
    B_noised = B.astype(np.int16) + E_B.astype(np.int16)

    searcher = JackpotSearcher(config, m, n)
    t0 = time.time()
    jackpot = searcher.search(A_noised, B_noised, backend=backend)
    elapsed = time.time() - t0

    # scan tiles for a hit
    hashes = 0
    for ri, ci, ro, co in searcher.tile_positions():
        h = jackpot_hash(list(jackpot[ri, ci]), a_seed)
        hashes += 1
        if max_hashes and hashes >= max_hashes:
            break
        if hash_le32(h) <= target:
            return _build_share(
                header, config, m, n, target, nonce, A, B,
                b_seed, a_seed, int(ro), int(co), h, jackpot[ri, ci],
                A_noised, B_noised, elapsed, hashes,
            )
    if on_attempt:
        on_attempt(nonce, elapsed, hashes)
    return None


def _build_share(header, config, m, n, target, nonce, A, B,
                 b_seed, a_seed, t_rows, t_cols, jackpot_hash_bytes,
                 jackpot_words, A_noised, B_noised, elapsed, hashes):
    """Assemble the self-contained share payload (documented 'v2' format)."""
    from .commitment import job_key
    from .merkle import MerkleTree

    key = job_key(header, config)
    k = config.common_dim

    a_tree = MerkleTree(A.astype(np.uint8, copy=False).tobytes(), key)
    b_tree = MerkleTree(B.T.astype(np.uint8, copy=False).tobytes(), key)

    rows = config.rows_pattern.indices_with_offset(t_rows)
    cols = config.cols_pattern.indices_with_offset(t_cols)

    a_proof = a_tree.proof_for_rows(rows, row_bytes=k, total_rows=m)
    b_proof = b_tree.proof_for_rows(cols, row_bytes=k, total_rows=n)

    return {
        "version": 2,
        "job_id": None,          # filled by the miner when a job is attached
        "nonce": f"{int(nonce):08x}",
        "header": header.to_bytes().hex(),
        "m": m, "n": n, "k": k, "rank": config.rank,
        "rows_pattern": config.rows_pattern.to_list(),
        "cols_pattern": config.cols_pattern.to_list(),
        "t_rows": t_rows,
        "t_cols": t_cols,
        "sigma": a_seed.hex(),
        "b_seed": b_seed.hex(),
        "target": f"{target:064x}",
        "jackpot_hash": jackpot_hash_bytes.hex(),
        "jackpot": [int(w) for w in jackpot_words],
        "a_root": hash_a_hex(key, A),
        "b_root": hash_b_hex(key, B),
        "a_tree_root": a_tree.root.hex(),
        "b_tree_root": b_tree.root.hex(),
        "a_proof": a_proof,
        "b_proof": b_proof,
        "elapsed_s": round(elapsed, 3),
        "hashes": hashes,
    }


def hash_a_hex(key: bytes, A: np.ndarray) -> str:
    from .commitment import pad_to_chunk_boundary
    import blake3 as _b3
    data = pad_to_chunk_boundary(A.astype(np.uint8, copy=False).tobytes())
    return _b3.blake3(data, key=key).digest().hex()


def hash_b_hex(key: bytes, B: np.ndarray) -> str:
    from .commitment import pad_to_chunk_boundary
    import blake3 as _b3
    data = pad_to_chunk_boundary(B.T.astype(np.uint8, copy=False).tobytes())
    return _b3.blake3(data, key=key).digest().hex()


def share_to_line(share: dict, rig: str, job_id: str) -> str:
    out = dict(share)
    out["rig"] = rig
    out["job_id"] = job_id
    return json.dumps(out, separators=(",", ":"))
