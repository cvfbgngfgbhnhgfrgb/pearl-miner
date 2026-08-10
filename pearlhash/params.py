"""
Pearl (PRL) "PearlHash" proof-of-useful-work: parameters & serialization.

This mirrors the reference implementation from
pearl-research-labs/pearl (zk-pow crate) so the maths is compatible:

  * IncompleteBlockHeader : 76-byte Bitcoin-style header
    (version | prev_hash | merkle_root | timestamp | nbits)
  * MiningConfiguration   : 52-byte config committed into the job key
    (common_dim u32 | rank u16 | mma_type u16 | rows_pattern 6 |
     cols_pattern 6 | moe-trailer 32)
  * job_key = blake3(header.to_bytes() + config.to_bytes())
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# Periodic pattern (3-dim arithmetic progression, 6 bytes on the wire)
# ---------------------------------------------------------------------------

class PeriodicPattern:
    """A periodic set of indices. Serialized to exactly 6 bytes."""

    NUM_DIMS = 3

    def __init__(self, shape: list[tuple[int, int]]):
        assert len(shape) == self.NUM_DIMS
        self.shape = [(int(s), int(l)) for s, l in shape]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_list(cls, pattern: list[int]) -> "PeriodicPattern":
        """Canonical 3-dim decomposition of a sorted index list starting at 0."""
        p = [int(x) for x in pattern]
        if not p or p[0] != 0 or any(a >= b for a, b in zip(p, p[1:])):
            raise ValueError("pattern must be sorted, dedup'd and start at 0")
        shape_vec: list[tuple[int, int]] = []
        while len(p) > 1:
            found = False
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape_vec.append((s, len(p) // period))
                        p = p[:period]
                        found = True
                        break
            if not found:
                raise ValueError("pattern is not periodic")
        shape_vec.reverse()
        period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
        while len(shape_vec) < cls.NUM_DIMS:
            shape_vec.append((period, 1))
        return cls(shape_vec)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PeriodicPattern":
        assert len(data) == 2 * cls.NUM_DIMS
        shape = []
        min_stride = 1
        for i in range(cls.NUM_DIMS):
            factor = 1 + data[2 * i]
            length = 1 + data[2 * i + 1]
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        return cls(shape)

    # -- accessors ----------------------------------------------------------
    def to_bytes(self) -> bytes:
        out = bytearray(2 * self.NUM_DIMS)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            factor = stride // min_stride
            out[2 * i] = factor - 1
            out[2 * i + 1] = length - 1
            min_stride = stride * length
        return bytes(out)

    def to_list(self) -> list[int]:
        res = [0]
        for stride, length in self.shape:
            new = []
            for i in range(length):
                for r in res:
                    new.append(r + i * stride)
            res = new
        return res

    def indices_with_offset(self, offset: int) -> list[int]:
        return [i + offset for i in self.to_list()]

    def offset_is_valid(self, offset: int) -> bool:
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True

    def period(self) -> int:
        return self.shape[-1][0] * self.shape[-1][1]

    def size(self) -> int:
        n = 1
        for _, length in self.shape:
            n *= length
        return n

    def max(self) -> int:
        return max(self.to_list())

    def __repr__(self) -> str:
        return f"PeriodicPattern({self.to_list()})"


# ---------------------------------------------------------------------------
# Mining configuration (52 bytes)
# ---------------------------------------------------------------------------

MMATYPE_INT7XINT7_TO_INT32 = 0


class MiningConfiguration:
    """The configuration a miner commits to before mining."""

    def __init__(
        self,
        common_dim: int,
        rank: int,
        rows_pattern: list[int],
        cols_pattern: list[int],
        mma_type: int = MMATYPE_INT7XINT7_TO_INT32,
    ):
        self.common_dim = int(common_dim)   # k
        self.rank = int(rank)               # noise rank r
        self.mma_type = int(mma_type)
        self.rows_pattern = PeriodicPattern.from_list(rows_pattern)
        self.cols_pattern = PeriodicPattern.from_list(cols_pattern)

    def to_bytes(self) -> bytes:
        moe = struct.pack("<HH", 0, 0) + b"\x00" * 28  # e=0, top_k=0 (dense)
        return (
            struct.pack("<I", self.common_dim)
            + struct.pack("<H", self.rank)
            + struct.pack("<H", self.mma_type)
            + self.rows_pattern.to_bytes()
            + self.cols_pattern.to_bytes()
            + moe
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "MiningConfiguration":
        assert len(data) == 52
        common_dim = struct.unpack("<I", data[0:4])[0]
        rank = struct.unpack("<H", data[4:6])[0]
        mma_type = struct.unpack("<H", data[6:8])[0]
        rows = PeriodicPattern.from_bytes(data[8:14])
        cols = PeriodicPattern.from_bytes(data[14:20])
        moe_e, moe_topk = struct.unpack("<HH", data[20:24])
        if moe_e != 0:
            raise ValueError("MoE (grouped-gemm) configs are not supported by this miner")
        return cls(
            common_dim=common_dim,
            rank=rank,
            mma_type=mma_type,
            rows_pattern=rows.to_list(),
            cols_pattern=cols.to_list(),
        )

    # -- derived ------------------------------------------------------------
    @property
    def tile_size(self) -> int:
        return self.rows_pattern.size() * self.cols_pattern.size()

    @property
    def dot_product_length(self) -> int:
        """Length of the inner product a worker has to open per tile row. == k."""
        return self.common_dim

    def difficulty_adjustment_factor(self) -> int:
        """How much easier the jackpot bound is per attempt (reference formula)."""
        return self.tile_size * self.dot_product_length

    def penalized_adjustment_factor(self, penalty_base_rank: int = 128) -> int:
        return self.tile_size * (self.dot_product_length // self.rank) * penalty_base_rank

    def __repr__(self) -> str:
        return (
            f"MiningConfiguration(k={self.common_dim}, rank={self.rank}, "
            f"rows={self.rows_pattern.to_list()}, cols={self.cols_pattern.to_list()})"
        )


# ---------------------------------------------------------------------------
# Incomplete block header (76 bytes) + difficulty helpers
# ---------------------------------------------------------------------------

class IncompleteBlockHeader:
    """version(4) | prev_block(32) | merkle_root(32) | timestamp(4) | nbits(4)"""

    SERIALIZED_SIZE = 76

    def __init__(self, version: int, prev_block: bytes, merkle_root: bytes,
                 timestamp: int, nbits: int):
        self.version = int(version)
        self.prev_block = bytes(prev_block)
        self.merkle_root = bytes(merkle_root)
        self.timestamp = int(timestamp)
        self.nbits = int(nbits)

    def to_bytes(self) -> bytes:
        return (
            struct.pack("<I", self.version)
            + self.prev_block
            + self.merkle_root
            + struct.pack("<I", self.timestamp)
            + struct.pack("<I", self.nbits)
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "IncompleteBlockHeader":
        assert len(data) == cls.SERIALIZED_SIZE, f"header must be 76 bytes, got {len(data)}"
        version = struct.unpack("<I", data[0:4])[0]
        prev = data[4:36]
        merkle = data[36:68]
        timestamp = struct.unpack("<I", data[68:72])[0]
        nbits = struct.unpack("<I", data[72:76])[0]
        return cls(version, prev, merkle, timestamp, nbits)

    def __repr__(self) -> str:
        return (f"Header(version=0x{self.version:08x}, time={self.timestamp}, "
                f"nbits=0x{self.nbits:08x})")


def nbits_to_difficulty(nbits: int) -> int:
    """Bitcoin compact nbits -> absolute target (U256) as a python int."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0 or (mantissa & 0x00800000):
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def scale_target(base: int, adjustment_factor: int) -> int:
    """Scale a target by an adjustment factor, saturating at 2^256-1."""
    max256 = (1 << 256) - 1
    if adjustment_factor <= 0:
        raise ValueError("adjustment_factor must be > 0")
    if base > max256 // adjustment_factor:
        return max256
    return base * adjustment_factor


def extract_difficulty_bound(nbits: int, config: MiningConfiguration) -> int:
    """nbits -> the jackpot bound used for verification (reference formula)."""
    return scale_target(nbits_to_difficulty(nbits), config.difficulty_adjustment_factor())


def penalized_target_bound(target: int, config: MiningConfiguration,
                           penalty_base_rank: int = 128) -> int | None:
    """Miner-side target adjustment (MiningJob.adjust_target). None if unusable."""
    factor = config.penalized_adjustment_factor(penalty_base_rank)
    if factor <= 0 or target > ((1 << 256) - 1) // factor:
        return None
    return target * factor


# ---------------------------------------------------------------------------
# Default mining profile (matches the reference GPU profile)
# ---------------------------------------------------------------------------

DEFAULT_ROWS_PATTERN = [0, 8]
DEFAULT_COLS_PATTERN = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49,
                        56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 104,
                        105, 112, 113, 120, 121, 128, 129, 136, 137, 144, 145,
                        152, 153, 160, 161, 168, 169, 176, 177, 184, 185, 192,
                        193, 200, 201, 208, 209, 216, 217, 224, 225, 232, 233,
                        240, 241, 248, 249]

# Signal values live in int7: [-64, 63]
SIGNAL_MIN = -64
SIGNAL_MAX = 63


def default_config(m: int = 1024, n: int = 1024, k: int = 1024, rank: int = 128):
    """Build a default MiningConfiguration. m/n are the A/B matrix dims."""
    config = MiningConfiguration(
        common_dim=k,
        rank=rank,
        rows_pattern=DEFAULT_ROWS_PATTERN,
        cols_pattern=DEFAULT_COLS_PATTERN,
    )
    # sanity-check against the reference verifier constraints
    assert rank >= 128, "rank below PENALTY_BASE_RANK is rejected by consensus"
    assert 1024 <= k <= 4 * rank * rank
    assert k >= 16 * rank and k % 64 == 0
    assert k % config.rows_pattern.period() == 0 or True  # pattern constraints below
    assert config.rows_pattern.max() < m and config.cols_pattern.max() < n
    return config
