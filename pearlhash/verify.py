"""
Share verifier (used by the mock pool and self-tests).

Given a share payload it:
  1. rebuilds header + mining config from the payload,
  2. recomputes job_key and checks sigma/b_seed against a_root/b_root,
  3. verifies the merkle paths of the opened rows/cols,
  4. re-derives the noise for the opened rows/cols, recomputes the tile
     jackpot and checks it equals the claimed jackpot_hash and passes
     the target.
"""

from __future__ import annotations

import base64
import struct

import blake3 as _blake3
import numpy as np

from .commitment import hash_le32, job_key, pad_to_chunk_boundary
from .mining import JackpotSearcher, rotl32
from .noise import NoiseGenerator
from .params import IncompleteBlockHeader, MiningConfiguration


class VerifyError(Exception):
    pass


def _check(cond: bool, msg: str):
    if not cond:
        raise VerifyError(msg)


def verify_share(share: dict) -> dict:
    """Return {'ok': True, 'details': ...} or raise VerifyError."""
    header = IncompleteBlockHeader.from_bytes(bytes.fromhex(share["header"]))
    config = MiningConfiguration(
        common_dim=share["k"],
        rank=share["rank"],
        rows_pattern=share["rows_pattern"],
        cols_pattern=share["cols_pattern"],
    )
    m, n, k = share["m"], share["n"], share["k"]

    # ---- 1. commitment --------------------------------------------------
    key = job_key(header, config)
    a_root = bytes.fromhex(share["a_root"])
    b_root = bytes.fromhex(share["b_root"])
    b_seed = bytes.fromhex(share["b_seed"])
    sigma = bytes.fromhex(share["sigma"])

    b_seed_hat = _blake3.blake3(key + b_root).digest()
    a_seed_hat = _blake3.blake3(b_seed_hat + a_root).digest()
    _check(b_seed_hat == b_seed, "b_seed mismatch")
    _check(a_seed_hat == sigma, "sigma (a_seed) mismatch")

    # ---- 2. merkle paths ------------------------------------------------
    rows = config.rows_pattern.indices_with_offset(share["t_rows"])
    cols = config.cols_pattern.indices_with_offset(share["t_cols"])
    _check(share["t_rows"] + config.rows_pattern.max() < m, "t_rows out of range")
    _check(share["t_cols"] + config.cols_pattern.max() < n, "t_cols out of range")

    a_tree_root = bytes.fromhex(share["a_tree_root"])
    b_tree_root = bytes.fromhex(share["b_tree_root"])
    _check_merkle_path(share["a_proof"], a_tree_root, key, "A")
    _check_merkle_path(share["b_proof"], b_tree_root, key, "B")

    # ---- 3. reconstruct opened rows/cols ---------------------------------
    A_rows = _extract_rows(share["a_proof"], rows, k)
    B_cols = _extract_rows(share["b_proof"], cols, k)   # rows of B^T == cols of B

    # ---- 4. noise for the opened rows/cols --------------------------------
    gen = NoiseGenerator(noise_rank=config.rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate_noise_matrices(
        key_A=sigma, key_B=b_seed,
        A_rows=m, common_dim=k, B_cols=n,
    )
    E_A = (E_AL.astype(np.int32) @ E_AR.astype(np.int32)).astype(np.int8)
    E_B = (E_BL.astype(np.int32) @ E_BR.astype(np.int32)).astype(np.int8)

    A_noised = A_rows.astype(np.int16) + E_A[rows].astype(np.int16)
    # B_noised[:, c] = B[:, c] + E_B[:, c]  ->  in B^T terms:
    # B_noised_t[c, :] = B_cols[c, :] + E_B[:, c]
    B_noised_t = B_cols.astype(np.int16) + E_B[:, cols].T.astype(np.int16)

    # ---- 5. recompute jackpot for the single tile --------------------------
    jackpot = _tile_jackpot(A_noised, B_noised_t, config)
    h = _blake3.blake3(
        b"".join(struct.pack("<I", int(w) & 0xFFFFFFFF) for w in jackpot),
        key=sigma,
    ).digest()

    _check(h.hex() == share["jackpot_hash"], "jackpot hash mismatch")

    target = int(share["target"], 16)
    _check(hash_le32(h) <= target, f"jackpot {hash_le32(h):064x} exceeds target {target:064x}")

    return {
        "ok": True,
        "t_rows": share["t_rows"],
        "t_cols": share["t_cols"],
        "jackpot_hash": h.hex(),
        "hash_le": hash_le32(h),
        "target": target,
    }


def _check_merkle_path(proof: dict, tree_root: bytes, key: bytes, label: str):
    leaves = {}
    for idx, data_b64 in zip(proof["leaf_indices"], proof["leaf_data"]):
        leaves[int(idx)] = data_b64
    for idx in proof["leaf_indices"]:
        idx = int(idx)
        chunk = base64.b64decode(leaves[idx])
        h = _blake3.blake3(chunk, key=key).digest()
        pos = idx
        sibs = proof["siblings"].get(str(idx), [])
        for sib_hex in sibs:
            sib = bytes.fromhex(sib_hex)
            if pos % 2 == 0:
                h = _blake3.blake3(h + sib, key=key).digest()
            else:
                h = _blake3.blake3(sib + h, key=key).digest()
            pos //= 2
        _check(h == tree_root, f"{label} merkle path for leaf {idx} does not reach root")


def _extract_rows(proof: dict, row_indices: list[int], row_bytes: int) -> np.ndarray:
    """Reconstruct the opened rows (int8) from the proof's raw chunks."""
    chunk_map = {int(i): base64.b64decode(d) for i, d in zip(proof["leaf_indices"], proof["leaf_data"])}
    rows = []
    for r in row_indices:
        start = r * row_bytes
        chunk_i = start // 1024
        off = start % 1024
        chunk = chunk_map[chunk_i]
        data = chunk[off:off + row_bytes]
        if len(data) < row_bytes:
            raise VerifyError(f"row {r} data incomplete")
        rows.append(np.frombuffer(data, dtype=np.uint8).astype(np.int8))
    return np.stack(rows)


def _tile_jackpot(A_noised: np.ndarray, B_noised_t: np.ndarray,
                  config: MiningConfiguration) -> list[int]:
    """Recompute the 16-word jackpot for a 2 x 64 tile."""
    rank = config.rank
    jackpot = [np.uint32(0)] * 16
    tile = np.zeros((A_noised.shape[0], B_noised_t.shape[0]), dtype=np.int32)
    num_chunks = config.common_dim // rank
    for p in range(num_chunks):
        ks = slice(p * rank, (p + 1) * rank)
        tile += A_noised[:, ks].astype(np.int32) @ B_noised_t[:, ks].astype(np.int32).T
        xored = np.bitwise_xor.reduce(tile.flatten().view(np.uint32))
        tid = p % 16
        jackpot[tid] = rotl32(jackpot[tid], 13) ^ xored
    return [int(w) for w in jackpot]
