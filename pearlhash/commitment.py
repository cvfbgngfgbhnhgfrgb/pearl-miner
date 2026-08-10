"""
Commitment hashing for PearlHash.

Reference (zk-pow/src/ffi/mine.rs):

    job_key   = blake3(header76 || config52)
    hash_a    = blake3(pad1024(A row-major),  key=job_key)
    hash_b    = blake3(pad1024(B^T col-major), key=job_key)
    b_seed    = blake3(job_key || hash_b)
    a_seed    = blake3(b_seed || hash_a)      ("sigma")

The jackpot PoW hash is keyed with a_seed:
    jackpot_hash = blake3(transcript64, key=a_seed)
"""

from __future__ import annotations

import struct

import blake3 as _blake3
import numpy as np

from .params import IncompleteBlockHeader, MiningConfiguration

BLAKE3_CHUNK = 1024


def pad_to_chunk_boundary(data: bytes) -> bytes:
    """Pad to the BLAKE3 chunk boundary (1024 bytes)."""
    rem = len(data) % BLAKE3_CHUNK
    if rem:
        data += b"\x00" * (BLAKE3_CHUNK - rem)
    return data


def job_key(header: IncompleteBlockHeader, config: MiningConfiguration) -> bytes:
    return _blake3.blake3(header.to_bytes() + config.to_bytes()).digest()


def commitment_seeds(header: IncompleteBlockHeader, config: MiningConfiguration,
                     A: np.ndarray, B: np.ndarray) -> tuple[bytes, bytes]:
    """Return (b_noise_seed, a_noise_seed) from int8 matrices A (m x k), B (k x n)."""
    key = job_key(header, config)
    a_row_major = pad_to_chunk_boundary(A.astype(np.uint8, copy=False).tobytes())
    b_col_major = pad_to_chunk_boundary(B.T.astype(np.uint8, copy=False).tobytes())
    hash_a = _blake3.blake3(a_row_major, key=key).digest()
    hash_b = _blake3.blake3(b_col_major, key=key).digest()
    b_seed = _blake3.blake3(key + hash_b).digest()
    a_seed = _blake3.blake3(b_seed + hash_a).digest()
    return b_seed, a_seed


def jackpot_hash(jackpot_words: list[int], a_seed: bytes) -> bytes:
    """blake3(16 x uint32 LE transcript, key=a_seed)."""
    msg = b"".join(struct.pack("<I", int(w) & 0xFFFFFFFF) for w in jackpot_words)
    return _blake3.blake3(msg, key=a_seed).digest()


def hash_le32(data: bytes) -> int:
    """Interpret a 32-byte digest as a little-endian 256-bit integer."""
    return int.from_bytes(data, "little")
