# -----------------------------------------------------------------------------
# Single source of truth for op schemas (Python side)
# -----------------------------------------------------------------------------
# - Mirrors C++ ops.hpp payloads
# - Vid[T] is always scalar-domain (no scalar flag)
# - Provides a registry OPS and helpers to derive read/write tables
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Generic, TypeVar
import torch
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
    i8 = 8

_TORCH_DTYPE_TO_DTYPEID: Dict[torch.dtype, DTypeId] = {
    torch.float16: DTypeId.f16,
    torch.float32: DTypeId.f32,
    torch.bfloat16: DTypeId.bf16,
    torch.int32: DTypeId.i32,
    torch.int64: DTypeId.i64,
    torch.uint32: DTypeId.u32,
    torch.uint8: DTypeId.u8,
    torch.int8: DTypeId.i8,
    torch.bool: DTypeId.boolean,
}

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
    x: Tid
    weight: Tid
    out: Tid
    bias: Optional[Tid]


@op("RMS_NORM")
@dataclass
class RMSNormNode:
    x: Tid
    weight: Tid
    out: Tid
    eps: float

@op("LAYER_NORM")
@dataclass
class LayerNormNode:
    x: Tid
    out: Tid
    weight: Optional[Tid]
    bias: Optional[Tid]
    eps: float

@op("ROPE_APPLY")
@dataclass
class RopeNode:
    q_in: Tid
    k_in: Tid
    q_out: Tid
    k_out: Tid
    head_dim: int
    pos: Vid[int]
    freqs: Optional[Tid]
    traditional: bool
    base: Optional[float]
    scale: float


@op("SDPA")
@dataclass
class SdpaNode:
    q: Tid
    k: Tid
    v: Tid
    out: Tid
    scale: float
    mask: Optional[Tid]
    causal: bool

@op("ADD")
@dataclass
class AddNode:
    a: Tid
    b: Tid
    out: Tid

@op("ADD_SCALAR")
@dataclass
class AddScalarNode:
    a: Union[int, Vid[int]]
    b: Union[int, Vid[int]]
    out: Vid[int]

@op("SYM_SIZE")
@dataclass
class SymSizeNode:
    a: Tid
    dim: int
    out: Vid[int]

@op("MUL")
@dataclass
class MulNode:
    a: Tid
    b: Tid
    out: Tid

@op("CONV_1D")
@dataclass
class Conv1DNode:
    x: Tid
    w: Tid
    out: Tid
    stride: int
    padding: int
    dilation: int
    groups: int

@op("GELU")
@dataclass
class GeluNode:
    x: Tid
    out: Tid

@op("ARANGE")
@dataclass
class ARangeNode:
    out: Tid
    start: int
    stop: int
    step: int
    dtype: Optional[DTypeId]

@op("SILU")
@dataclass
class SiluNode:
    x: Tid
    out: Tid

@op("RESHAPE")
@dataclass
class ReshapeNode:
    x: Tid
    out: Tid
    shape: List[Union[int, Vid[int]]]
    __meta__ = op_meta(view_of="x")


@op("TRANSPOSE")
@dataclass
class TransposeNode:
    x: Tid
    out: Tid
    perm: List[int]
    __meta__ = op_meta(view_of="x")


@op("CONTIGUOUS")
@dataclass
class ContigNode:
    x: Tid
    out: Tid

@op("ID_COPY")
@dataclass
class IdCopyNode:
    x: Tid
    out: Tid

@op("GATHER")
@dataclass
class GatherNode:
    table: Tid
    ids: Tid
    out: Tid

@op("SLICE")
@dataclass
class SliceNode:
    x: Tid
    out: Tid
    axis: Union[int, Vid[int]]
    start: Union[int, Vid[int]]
    end: Union[int, Vid[int]]

@op("CAST")
@dataclass
class CastNode:
    x: Tid
    out: Tid
    dtype: DTypeId


@op("QUANTIZED_LINEAR")
@dataclass
class QuantizedLinearNode:
    x: Tid
    w: Tid
    scales: Tid
    out: Tid
    biases: Optional[Tid]
    bias: Optional[Tid]
    group_size: int
    bits: int
    mode: str
    out_dtype: DTypeId


@op("CONCAT")
@dataclass
class ConcatNode:
    a: Tid
    b: Tid
    out: Tid
    axis: int


@op("FULL")
@dataclass
class FullNode:
    out: Tid
    shape: List[int]
    v: float
    dtype: DTypeId


@op("ZEROS")
@dataclass
class ZerosNode:
    out: Tid
    shape: List[int]
    dtype: DTypeId


@op("ONES")
@dataclass
class OnesNode:
    out: Tid
    shape: List[int]
    dtype: DTypeId


@op("ARGMAX")
@dataclass
class ArgmaxNode:
    x: Tid
    out: Tid
    axis: int


@op("SLICE_UPDATE")
@dataclass
class SliceUpdateNode:
    dst: Tid
    update: Tid
    axis: Union[int, Vid[int]]
    start: Union[int, Vid[int]]
    stop: Union[int, Vid[int]]


@op("QUANTIZED_GATHER")
@dataclass
class QuantizedGatherNode:
    table_q: Tid
    scales: Tid
    ids: Tid
    out: Tid
    biases: Optional[Tid]
    group_size: int
    bits: int
    mode: str
    out_dtype: DTypeId
