"""
Chunked keyed-BLAKE3 Merkle tree used for opened-row/col proofs.

NOTE (compatibility): the reference pearl-blake3 `MerkleTree` uses BLAKE3's
*native* tree layout, where the tree root equals the single-shot keyed hash of
the whole padded blob. This module implements a *conventional* balanced binary
Merkle tree over 1024-byte chunks, which is simpler to implement & verify.
The noise-seed commitment (which is what consensus actually checks) still uses
the single-shot `blake3(padded, key)` exactly like the reference, so shares are
mathematically consistent. If you need byte-exact reference proofs, swap in the
official `pearl_mining` Rust bindings for this class.
"""

from __future__ import annotations

import base64
import math

import blake3 as _blake3

from .commitment import BLAKE3_CHUNK, pad_to_chunk_boundary


def _leaf_hash(chunk: bytes, key: bytes) -> bytes:
    return _blake3.blake3(chunk, key=key).digest()


def _node_hash(left: bytes, right: bytes, key: bytes) -> bytes:
    return _blake3.blake3(left + right, key=key).digest()


class MerkleTree:
    """Balanced binary keyed merkle tree over 1024-byte chunks."""

    def __init__(self, data: bytes, key: bytes, chunk_size: int = BLAKE3_CHUNK):
        self.key = key
        self.chunk_size = chunk_size
        padded = pad_to_chunk_boundary(data)
        assert len(padded) % chunk_size == 0
        self._chunks = [
            padded[i:i + chunk_size] for i in range(0, len(padded), chunk_size)
        ]
        self.leaves = [_leaf_hash(c, key) for c in self._chunks]
        self.root = self._build(self.leaves)

    def _build(self, level: list[bytes]) -> bytes:
        if len(level) == 1:
            return level[0]
        if len(level) % 2 == 1:
            level = level + [level[-1]]  # duplicate last leaf
        nxt = [
            _node_hash(level[i], level[i + 1], self.key)
            for i in range(0, len(level), 2)
        ]
        return self._build(nxt)

    @staticmethod
    def compute_leaf_index(row: int, row_bytes: int, chunk_size: int = BLAKE3_CHUNK) -> int:
        """Leaf index covering the byte range of a matrix row."""
        start = row * row_bytes
        return start // chunk_size

    def proof_for_rows(self, row_indices: list[int], row_bytes: int,
                       total_rows: int) -> dict:
        """Build a proof for whole rows. Rows are contiguous byte ranges.

        Returns a dict with leaf indices, leaf data (raw chunk bytes), and
        sibling hashes along each leaf's path.
        """
        leaf_indices = sorted({
            self.compute_leaf_index(r, row_bytes) for r in row_indices
        })
        return self.proof_for_leaves(leaf_indices)

    def proof_for_leaves(self, leaf_indices: list[int]) -> dict:
        n = len(self.leaves)
        # pad leaf count to a power of two (mirrors tree construction)
        padded_n = 1 << max(0, (n - 1).bit_length())
        levels = [self.leaves + [self.leaves[-1]] * (padded_n - n)]
        while len(levels[-1]) > 1:
            lvl = levels[-1]
            nxt = [
                _node_hash(lvl[i], lvl[i + 1], self.key)
                for i in range(0, len(lvl), 2)
            ]
            levels.append(nxt)

        siblings: dict[int, list[bytes]] = {}
        for idx in leaf_indices:
            sibs = []
            pos = idx
            for lvl in levels[:-1]:
                peer = pos ^ 1
                sibs.append(lvl[peer] if peer < len(lvl) else lvl[-1])
                pos //= 2
            siblings[idx] = sibs

        return {
            "leaf_indices": leaf_indices,
            "leaf_data": [base64.b64encode(self._chunks[i]).decode() for i in leaf_indices],
            "siblings": {str(i): [s.hex() for s in sibs] for i, sibs in siblings.items()},
            "root": self.root.hex(),
            "total_leaves": n,
        }
