"""
Noise generation for the PearlHash noisy GEMM.

Faithful port of miner/miner-base `noise_generation.py` (official repo):

  E_A = E_AL @ E_AR     (m x k, int8)  -> added to A
  E_B = E_BL @ E_BR     (k x n, int8)  -> added to B

where E_AL, E_BR are uniform random low-rank matrices and E_AR, E_BL are
permutation matrices (+1 / -1), all derived from keyed BLAKE3 draws seeded
by the commitment (a_noise_seed / b_noise_seed).
"""

from __future__ import annotations

import math
import struct

import blake3 as _blake3

# A torch-free implementation; callers pass numpy arrays.
import numpy as np


class NoiseGenerator:
    def __init__(self, noise_rank: int = 128, noise_range: int = 128):
        if not (noise_range and (noise_range & (noise_range - 1)) == 0):
            raise ValueError("noise_range must be a power of two")
        if not (noise_rank and (noise_rank & (noise_rank - 1)) == 0):
            raise ValueError("noise_rank must be a power of two")
        if noise_range > 128:
            raise ValueError("noise_range must fit in uint7")
        if noise_rank % _blake3.blake3().digest_size != 0:
            raise ValueError("noise_rank must be divisible by 32")

        self.noise_rank = noise_rank
        self.noise_range = noise_range

        idxs_per_col = 2
        _noise_range = noise_range // idxs_per_col
        self.zero_point_translation = _noise_range // 2
        self.range_mask = _noise_range - 1
        self.rank_mask = noise_rank - 1

    # ------------------------------------------------------------------ #
    def generate_noise_matrices(
        self,
        key_A: bytes,
        key_B: bytes,
        A_rows: int,
        common_dim: int,
        B_cols: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (E_AL, E_AR, E_BL, E_BR) int8 arrays."""
        seed_A = b"A_tensor" + b"\x00" * 24
        seed_B = b"B_tensor" + b"\x00" * 24

        A_L = self._uniform_random_matrix(seed_A, key_A, A_rows)                 # m x r
        A_R = self._permutation_matrix(seed_A, key_A, self.noise_rank,
                                       common_dim, assign_columns=True)          # r x k
        B_L = self._permutation_matrix(seed_B, key_B, common_dim,
                                       self.noise_rank, assign_columns=False)    # k x r
        B_R = self._uniform_random_matrix(seed_B, key_B, B_cols).T               # r x n
        return A_L, A_R, B_L, B_R

    # ------------------------------------------------------------------ #
    def _get_random_hash(self, index: int, seed: bytes, key: bytes,
                         prepend_index: int) -> bytes:
        message_prepend = np.zeros(8, dtype=np.int32)
        message_prepend[prepend_index] = 1 + index
        message_bytes = message_prepend.tobytes() + seed
        return _blake3.blake3(message_bytes, key=key).digest()

    def _uniform_random_matrix(self, seed: bytes, key: bytes, rows: int) -> np.ndarray:
        cols = self.noise_rank
        draws = int(math.ceil(rows * cols / _blake3.blake3().digest_size))
        random_bytes = b"".join(
            self._get_random_hash(i, seed, key, 0) for i in range(draws)
        )[: rows * cols]
        arr = np.frombuffer(random_bytes, dtype=np.uint8).astype(np.int32)
        noise = ((arr & self.range_mask) - self.zero_point_translation).astype(np.int8)
        return noise.reshape(rows, cols)

    def _permutation_matrix(self, seed: bytes, key: bytes, rows: int, cols: int,
                            assign_columns: bool) -> np.ndarray:
        assert rows == self.noise_rank or cols == self.noise_rank
        noise_matrix = np.zeros((rows, cols), dtype=np.int8)
        if assign_columns:
            required_lines = cols
            assert rows == self.noise_rank
        else:
            required_lines = rows
            assert cols == self.noise_rank

        bytes_per_lines = 4
        draws = int(math.ceil(required_lines * bytes_per_lines / _blake3.blake3().digest_size))

        for i in range(draws):
            h = self._get_random_hash(i, seed, key, 1)
            u32s = np.frombuffer(h, dtype=np.uint32)
            for k in range(_blake3.blake3().digest_size // bytes_per_lines):
                assignment_index = i * (_blake3.blake3().digest_size // bytes_per_lines) + k
                if assignment_index >= required_lines:
                    break
                r = u32s[k]
                first_idx = int(r) & self.rank_mask
                second_idx = first_idx ^ (1 + int(np_mul_hi_u32(
                    np.uint32(self.noise_rank - 1), r)))
                perm = np.zeros(self.noise_rank, dtype=np.int8)
                perm[first_idx] = 1
                perm[second_idx] = -1
                if assign_columns:
                    noise_matrix[:, assignment_index] = perm
                else:
                    noise_matrix[assignment_index, :] = perm
        return noise_matrix


def np_mul_hi_u32(a: np.uint32, b: np.uint32) -> np.uint32:
    prod64 = np.uint64(a) * np.uint64(b)
    return np.uint32(prod64 >> np.uint64(32))


def add_noise(A: np.ndarray, B: np.ndarray, seeds, rank: int = 128):
    """Add noise to int8 matrices A (m x k) and B (k x n).

    seeds = (b_noise_seed, a_noise_seed)
    Returns (A_noised int16, B_noised int16).
    """
    b_seed, a_seed = seeds
    gen = NoiseGenerator(noise_rank=rank)
    E_AL, E_AR, E_BL, E_BR = gen.generate_noise_matrices(
        key_A=a_seed, key_B=b_seed,
        A_rows=A.shape[0], common_dim=A.shape[1], B_cols=B.shape[1],
    )

    E_A = (E_AL.astype(np.int32) @ E_AR.astype(np.int32)).astype(np.int8)
    E_B = (E_BL.astype(np.int32) @ E_BR.astype(np.int32)).astype(np.int8)

    A_noised = A.astype(np.int16) + E_A.astype(np.int16)
    B_noised = B.astype(np.int16) + E_B.astype(np.int16)
    return A_noised, B_noised
