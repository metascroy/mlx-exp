# -----------------------------------------------------------------------------
# Single source of truth for op schemas (Python side)
# -----------------------------------------------------------------------------
# - Mirrors C++ ops.hpp payloads
# - Uses dataclasses + field(metadata=...) to tag READ/WRITE/ATTR roles
# - Vid[T] is always scalar-domain (no scalar flag)
# - Provides a registry OPS and helpers to derive read/write tables
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Generic, TypeVar

# -----------------------------------------------------------------------------
# Core slot types for codegen
# -----------------------------------------------------------------------------

T = TypeVar("T")

class Tid:
    """Tensor identifier (index into tensor slots)."""
    pass

class Vid(Generic[T]):
    """Value identifier parameterized by Python scalar type for codegen."""
    pass

# -----------------------------------------------------------------------------
# Role tags / helpers
# -----------------------------------------------------------------------------

class PortRole:
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"
    ATTR = "attr"  # non-slot attribute
    MAYBE_READ = "maybe_read"


def port(role: str, *, optional: bool = False) -> Dict[str, Any]:
    """Attach to dataclass field metadata."""
    return {"role": role, "optional": optional}

# -----------------------------------------------------------------------------
# Op registry + meta helpers
# -----------------------------------------------------------------------------

OPS: Dict[str, Type] = {}

def op(name: str):
    """Decorator to register an op dataclass under OPS[name]."""
    def deco(cls):
        OPS[name] = cls
        cls.__opname__ = name
        return cls
    return deco

def op_meta(**kw) -> Dict[str, Any]:
    return {"op_meta": kw}

# -----------------------------------------------------------------------------
# DType enum (Python side mirror of C++ DTypeId)
# -----------------------------------------------------------------------------

class DTypeId(Enum):
    f16 = 0
    f32 = 1
    bf16 = 2
    i32 = 3
    i64 = 4
    u32 = 5
    u8 = 6
    boolean = 7

# -----------------------------------------------------------------------------
# Op definitions — mirror of C++ payload structs
# -----------------------------------------------------------------------------

@op("NOOP")
@dataclass
class NoopNode:
    pass


@op("LINEAR")
@dataclass
class LinearNode:
    x: Tid = field(metadata=port(PortRole.READ))
    weight: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    bias: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))


@op("RMS_NORM")
@dataclass
class RMSNormNode:
    x: Tid = field(metadata=port(PortRole.READ))
    weight: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    eps: float = field(default=1e-5, metadata=port(PortRole.ATTR))


@op("ROPE_APPLY")
@dataclass
class RopeNode:
    q_in: Tid = field(metadata=port(PortRole.READ))
    k_in: Tid = field(metadata=port(PortRole.READ))
    q_out: Tid = field(metadata=port(PortRole.WRITE))
    k_out: Tid = field(metadata=port(PortRole.WRITE))
    head_dim: int = field(metadata=port(PortRole.ATTR))
    pos: Vid[int] = field(metadata=port(PortRole.READ))
    freqs: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))
    traditional: bool = field(default=False, metadata=port(PortRole.ATTR))
    base: Optional[float] = field(default=500000.0, metadata=port(PortRole.ATTR, optional=True))
    scale: float = field(default=1.0, metadata=port(PortRole.ATTR))


@op("SDPA")
@dataclass
class SdpaNode:
    q: Tid = field(metadata=port(PortRole.READ))
    k: Tid = field(metadata=port(PortRole.READ))
    v: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    scale: float = field(default=1.0, metadata=port(PortRole.ATTR))
    mask: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))
    causal: bool = field(default=False, metadata=port(PortRole.ATTR))


@op("ADD")
@dataclass
class AddNode:
    a: Tid = field(metadata=port(PortRole.READ))
    b: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))


@op("MUL")
@dataclass
class MulNode:
    a: Tid = field(metadata=port(PortRole.READ))
    b: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))


@op("SILU")
@dataclass
class SiluNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))


@op("RESHAPE")
@dataclass
class ReshapeNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    shape: List[int] = field(metadata=port(PortRole.ATTR))
    __meta__ = op_meta(view_of="x")


@op("TRANSPOSE")
@dataclass
class TransposeNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    perm: List[int] = field(metadata=port(PortRole.ATTR))
    __meta__ = op_meta(view_of="x")


@op("CONTIGUOUS")
@dataclass
class ContigNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))


@op("GATHER")
@dataclass
class GatherNode:
    table: Tid = field(metadata=port(PortRole.READ))
    ids: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))


@op("SLICE")
@dataclass
class SliceNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    axis: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))
    start: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))
    length: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))


@op("CAST")
@dataclass
class CastNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    dtype: DTypeId = field(default=DTypeId.f16, metadata=port(PortRole.ATTR))


@op("QUANTIZED_LINEAR")
@dataclass
class QuantizedLinearNode:
    x: Tid = field(metadata=port(PortRole.READ))
    w: Tid = field(metadata=port(PortRole.READ))
    scales: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))  # <-- move before defaults
    biases: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))
    bias: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))
    group_size: int = field(default=64, metadata=port(PortRole.ATTR))
    bits: int = field(default=4, metadata=port(PortRole.ATTR))
    mode: str = field(default="affine", metadata=port(PortRole.ATTR))
    out_dtype: DTypeId = field(default=DTypeId.f32, metadata=port(PortRole.ATTR))


@op("CONCAT")
@dataclass
class ConcatNode:
    a: Tid = field(metadata=port(PortRole.READ))
    b: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    axis: int = field(default=0, metadata=port(PortRole.ATTR))


@op("FULL")
@dataclass
class FullNode:
    out: Tid = field(metadata=port(PortRole.WRITE))
    shape: List[int] = field(metadata=port(PortRole.ATTR))
    v: float = field(default=0.0, metadata=port(PortRole.ATTR))
    dtype: DTypeId = field(default=DTypeId.f16, metadata=port(PortRole.ATTR))


@op("ZEROS")
@dataclass
class ZerosNode:
    out: Tid = field(metadata=port(PortRole.WRITE))
    shape: List[int] = field(metadata=port(PortRole.ATTR))
    dtype: DTypeId = field(default=DTypeId.f16, metadata=port(PortRole.ATTR))


@op("ONES")
@dataclass
class OnesNode:
    out: Tid = field(metadata=port(PortRole.WRITE))
    shape: List[int] = field(metadata=port(PortRole.ATTR))
    dtype: DTypeId = field(default=DTypeId.f16, metadata=port(PortRole.ATTR))


@op("ARGMAX")
@dataclass
class ArgmaxNode:
    x: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    axis: int = field(default=-1, metadata=port(PortRole.ATTR))


@op("SLICE_UPDATE")
@dataclass
class SliceUpdateNode:
    dst: Tid = field(metadata=port(PortRole.READWRITE))
    update: Tid = field(metadata=port(PortRole.READ))
    axis: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))
    start: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))
    length: Union[int, Vid[int]] = field(metadata=port(PortRole.MAYBE_READ))


@op("QUANTIZED_GATHER")
@dataclass
class QuantizedGatherNode:
    table_q: Tid = field(metadata=port(PortRole.READ))
    scales: Tid = field(metadata=port(PortRole.READ))
    ids: Tid = field(metadata=port(PortRole.READ))
    out: Tid = field(metadata=port(PortRole.WRITE))
    biases: Optional[Tid] = field(default=None, metadata=port(PortRole.READ, optional=True))
    group_size: int = field(default=64, metadata=port(PortRole.ATTR))
    bits: int = field(default=4, metadata=port(PortRole.ATTR))
    mode: str = field(default="affine", metadata=port(PortRole.ATTR))
    out_dtype: DTypeId = field(default=DTypeId.f32, metadata=port(PortRole.ATTR))

# -----------------------------------------------------------------------------
# Derived tables (for codegen / liveness / validators)
# -----------------------------------------------------------------------------

def derive_reads_writes():
    rw = {}
    for opname, cls in OPS.items():
        rnames, wnames = [], []
        for f in fields(cls):
            role = f.metadata.get("role")
            if role in (PortRole.READ, PortRole.READWRITE, PortRole.MAYBE_READ):
                rnames.append(f.name)
            if role in (PortRole.WRITE, PortRole.READWRITE):
                wnames.append(f.name)
        rw[opname] = (tuple(rnames), tuple(wnames))
    return rw


def view_sources() -> Dict[str, str]:
    """Return {opname: input_field_name_that_out_views} for true views."""
    res: Dict[str, str] = {}
    for opname, cls in OPS.items():
        meta = getattr(cls, "__meta__", None)
        if isinstance(meta, dict):
            mm = meta.get("op_meta", {})
            if "view_of" in mm:
                res[opname] = mm["view_of"]
    return res


READS_WRITES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = derive_reads_writes()
