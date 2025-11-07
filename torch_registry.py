from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx
from torch.fx.node import Node

from _program_builder import ProgramBuilder, REGISTRY, Slot, Handler, PatternHandler
from ops_schema import DTypeId, _TORCH_DTYPE_TO_DTYPEID
import operator
import torchao # to register dequantize_affine

@torch.library.custom_op("mlx::apply_rope", mutates_args=())
def _(
    q_in: torch.Tensor,  # (B, Hq, T, D)
    k_in: torch.Tensor,  # (B, Hk, T, D)
    head_dim: int,
    pos: int,                  # int, not tensor
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Dh = int(head_dim)
    assert q_in.size(-1) == Dh and k_in.size(-1) == Dh, "head_dim mismatch"

    # unpack as (B, H, T, D)
    B, Hq, T, _ = q_in.shape
    B2, Hk, T2, _ = k_in.shape
    assert B == B2 and T == T2, "RoPE expects q and k to have same B,T"
    half = Dh // 2

    if freqs is None:
        # [1, 1, 1, half] to broadcast over B,H,T
        i = torch.arange(half, device=q_in.device, dtype=torch.float32)
        inv_freq = (base ** (-2.0 * i / Dh)).view(1, 1, 1, half)

        # positions: [1, 1, T, 1]
        pos_range = torch.arange(
            pos, pos + T, device=q_in.device, dtype=torch.float32
        ).view(1, 1, T, 1)

        # final angles: [1, 1, T, half]
        angles = (pos_range * inv_freq) * float(scale)
    else:
        # assume freqs is already per-position, just reshape to [1,1,T,half]
        angles = freqs.to(torch.float32).view(1, 1, T, half)

    cos = angles.cos().to(q_in.dtype)  # [1,1,T,half]
    sin = angles.sin().to(q_in.dtype)  # [1,1,T,half]

    def rot(x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        x1, x2 = x[..., :half], x[..., half : 2 * half]
        xr = x1 * cos - x2 * sin
        xi = x1 * sin + x2 * cos
        if 2 * half != Dh:
            return torch.cat([xr, xi, x[..., 2 * half :]], dim=-1)
        return torch.cat([xr, xi], dim=-1)

    q_out = rot(q_in)
    k_out = rot(k_in)
    return q_out, k_out


@torch.library.register_fake("mlx::apply_rope")
def _(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    head_dim: int,
    pos: int,
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # just mirror the shapes
    return (
        q_in.new_empty(q_in.shape),
        k_in.new_empty(k_in.shape),
    )

@torch.library.custom_op("mlx::rms_norm", mutates_args=())
def _(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    y = y.to(x.dtype)
    return y * weight.to(x.dtype)


@torch.library.register_fake("mlx::rms_norm")
def _(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x.new_empty(x.shape)


@REGISTRY.register(target=[torch.ops.aten.embedding.default])
def embedding_handler(P: ProgramBuilder, n: Node) -> Slot:
    assert len(n.kwargs) == 0, f"Got unexpected kwargs={kwargs}"
    args = P.args(n)
    w, x = args[0], args[1]
    # print("EMBEDDING ARGS", n.args, n.kwargs)
    out = P.make_or_get_slot(n)
    P.GATHER(table=w, ids=x, out=out)
    return out

@REGISTRY.register(target=[torch.ops.mlx.rms_norm.default])
def rms_norm_handler(P: ProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    x, w = args[0], args[1]
    eps = 1e-5
    if len(args) >= 3:
        eps = args[2]
    out = P.make_or_get_slot(n)
    P.RMS_NORM(x=x, weight=w, out=out, eps=eps)
    # print("IN RMS NORM", "x", x, "w", w, "out", out)
    return out

@REGISTRY.register(target=[torch.ops.aten.view.default])
def view_handler(P: ProgramBuilder, n: Node) -> Slot:
    # print("VIEW ARGS", n.args, n.kwargs)
    x, shape = P.args(n)
    out = P.make_or_get_slot(n)
    P.RESHAPE(x=x, out=out, shape=shape)
    return out

@REGISTRY.register(target=[torch.ops.aten.clone.default])
def clone_handler(P: ProgramBuilder, n: Node) -> Slot:
    x, = P.args(n)
    out = P.make_or_get_slot(n)
    P.CONTIGUOUS(x=x, out=out)
    return out

@REGISTRY.register(target=[torch.ops.mlx.apply_rope.default])
def apply_rope_handler(P: ProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    q_in, k_in, head_dim, pos = args[0], args[1], args[2], args[3]
    traditional = args[4] if len(args) > 4 else False
    base = args[5] if len(args) > 5 else 500000.0
    scale = args[6] if len(args) > 6 else 1.0
    freqs = args[7] if len(args) > 7 else None
    out = P.make_or_get_slot(n)
    P.ROPE_APPLY(q_in=q_in, k_in=k_in, q_out=out[0], k_out=out[1], head_dim=head_dim, pos=pos, freqs=freqs, traditional=traditional, base=base, scale=scale)
    return out

@REGISTRY.register(target=[torch.ops.aten.add.Tensor])
def add_handler(P: ProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)
    out = P.make_or_get_slot(n)
    P.ADD(a=a, b=b, out=out)
    return out

@REGISTRY.register(target=[operator.add])
def add_handler(P: ProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)
    out = P.make_or_get_slot(n)
    P.ADD_SCALAR(a=a, b=b, out=out)
    return out

@REGISTRY.register(target=[torch.ops.aten.sym_size.int])
def sym_size_handler(P: ProgramBuilder, n: Node) -> Slot:
    a, dim = P.args(n)
    out = P.make_or_get_slot(n)
    P.SYM_SIZE(a=a, dim=dim, out=out)
    return out

@REGISTRY.register(target=["NOOP", torch.ops.aten._assert_scalar.default])
def no_op_handler(P: ProgramBuilder, n: Node) -> None:
    pass

@REGISTRY.register(target=[operator.getitem])
def getitem_handler(P: ProgramBuilder, n: Node) -> Slot:
    assert n.kwargs == {}, f"Got unexpected kwargs={kwargs}"
    a, idx = P.args(n)
    out = P.make_or_get_slot(n)
    P.ID_COPY(x=a[idx], out=out)
    return out

@REGISTRY.register(target=[torch.ops.aten.permute.default])
def permute_handler(P: ProgramBuilder, n: Node) -> Slot:
    x, dims = P.args(n)
    out = P.make_or_get_slot(n)
    P.TRANSPOSE(x=x, out=out, perm=dims)
    return out

@REGISTRY.register(target=[torch.ops.aten.slice.Tensor])
def slice_handler(P: ProgramBuilder, n: Node) -> Slot:
    # print("HANDLING SLICE", n.args, n.kwargs)
    x, dim, start, end = P.args(n)
    if start is None:
        start = 0
    # print("SLICE SLOTS", x, dim, start, end)
    out = P.make_or_get_slot(n)
    P.SLICE(x=x, out=out, axis=dim, start=start, end=end)
    return out

@REGISTRY.register(target=[torch.ops.aten.linear.default])
def linear_handler(P: ProgramBuilder, n: Node) -> Slot:
    assert len(n.kwargs) == 0, f"Got unexpected kwargs={kwargs}"
    args = P.args(n)
    x, w = args[0], args[1]
    b = None
    if len(args) > 2:
        b = args[2]
    out = P.make_or_get_slot(n)
    P.LINEAR(x=x, weight=w, out=out, bias=b)
    return out

@REGISTRY.register(target=[torch.ops.aten.silu.default])
def silu_handler(P: ProgramBuilder, n: Node) -> Slot:
    x = P.args(n)[0]
    out = P.make_or_get_slot(n)
    P.SILU(x=x, out=out)
    return out

@REGISTRY.register(target=[torch.ops.aten.mul.Tensor])
def mul_handler(P: ProgramBuilder, n: Node) -> Slot:
    a, b = P.args(n)
    out = P.make_or_get_slot(n)
    P.MUL(a=a, b=b, out=out)
    return out

@REGISTRY.register_pattern(name="SLICE_UPDATE")
class SliceUpdateHandler(PatternHandler):
    def __init__(self, head, body, dst, update, axis, start, stop):
        super().__init__(head, body)
        self.dst = dst
        self.update = update
        self.axis = axis
        self.start = start
        self.stop = stop
    
    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[SliceUpdateHandler]:
        _op_namespace = torch.ops.aten

        slice_scatter_node = head
        if slice_scatter_node.target !=_op_namespace.slice_scatter.default:
            return None

        # Slice scatter should write to a mutable input/buffer to be a slice update
        if ((slice_scatter_node.name not in ep.graph_signature.buffers_to_mutate) and 
           (slice_scatter_node.name not in ep.graph_signature.user_inputs_to_mutate)):
            return None

        if len(slice_scatter_node.args) != 5:
            return None
        ss_dst, ss_src, ss_axis, ss_start, ss_end = slice_scatter_node.args

        copy_node = ss_src
        if copy_node.target != _op_namespace.copy.default:
            return None
        if copy_node.users != {slice_scatter_node: None}:
            return None
        if len(copy_node.args) != 2:
            return None
        c_dst, c_src = copy_node.args

        slice_node = c_dst
        if slice_node.target != _op_namespace.slice.Tensor:
            return None
        if slice_node.users != {copy_node: None}:
            return None
        if len(slice_node.args) != 4:
            return None
        s_src, s_axis, s_start, s_end = slice_node.args

        # Slice should be on a buffer/input to be a slice-update
        if (
            (s_src.name not in ep.graph_signature.inputs_to_buffers) and 
            (s_src.name not in ep.graph_signature.user_inputs)
        ):
            return None
        
        # We should be slice / slice-scatter the same input/buffer in a slice update
        if s_src.name in ep.graph_signature.inputs_to_buffers:
            buf = ep.graph_signature.inputs_to_buffers[s_src.name]
            buf_mut = ep.graph_signature.buffers_to_mutate[slice_scatter_node.name]
            if buf != buf_mut:
                return None

        if s_src.name in ep.graph_signature.user_inputs:
            inp = ep.graph_signature.user_inputs[s_src.name]
            inp_mut = ep.graph_signature.user_inputs_to_mutate.get(slice_scatter_node.name, None)
            if inp != inp_mut:
                return None

        if (
            (s_src != ss_dst) or
            (s_axis != ss_axis) or
            (s_start != ss_start) or
            (s_end != ss_end)
        ):
            return None
        
        head = slice_scatter_node
        body = [slice_node, copy_node]
        dst = s_src
        update = c_src
        axis = s_axis
        start = s_start
        stop = s_end
        return SliceUpdateHandler(head, body, dst, update, axis, start, stop)
    
    def __call__(self, P: ProgramBuilder, n: Node):
        assert n == self.head
        kwargs = P.slot_map({k:getattr(self, k) for k in ["dst", "update", "axis", "start", "stop"]})
        P.SLICE_UPDATE(**kwargs)
        # print("RETURNING DST FROM SLICE UPDATE", kwargs["dst"])

        P.set_slot(n, kwargs["dst"])
        return kwargs["dst"]



@REGISTRY.register_pattern(name="SDPA")
class SDPAHandler(PatternHandler):
    def __init__(self, head, body, q_node, k_node, v_node):
        super().__init__(head, body)
        self.q_node = q_node
        self.k_node = k_node
        self.v_node = v_node
    
    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[SliceUpdateHandler]:
        _op_namespace = torch.ops.aten

        sdpa_node = head
        if sdpa_node.target !=_op_namespace.scaled_dot_product_attention.default:
            return None
        
        q, k, v, attn_mask, is_causal, scale = sdpa_node.args

        # Detect grouped kv attention pattern with repeat_interleave before SDPA
        is_grouped_kv = False
        k_base = k
        v_base = v
        if (
            (k.target is _op_namespace.repeat_interleave.self_int) and (k.users == {sdpa_node: None}) and (len(k.args) == 3) and (len(k.kwargs) == 0) and 
            (v.target is _op_namespace.repeat_interleave.self_int) and (v.users == {sdpa_node: None}) and (len(v.args) == 3) and (len(v.kwargs) == 0)
        ):
            k_unrepeated, k_reps, k_dim = k.args
            v_unrepeated, v_reps, v_dim = v.args

            if (k_dim == 1 and v_dim == 1) and (k_reps == v_reps):
                is_grouped_kv = True
                k_base = k_unrepeated
                v_base = v_unrepeated
        
        head = sdpa_node
        body = [k, v] if is_grouped_kv else []
        return SDPAHandler(head, body, q_node=q, k_node=k_base, v_node=v_base)
    
    def __call__(self, P: ProgramBuilder, n: Node):
        assert n == self.head
        # print("SDPA ARGS", n.args, n.kwargs)
        q, k, v, attn_mask, dropout_p, is_causal = n.args[0:7]
        enable_gqa = n.args[7] if len(n.args) > 7 else False
        scale = n.kwargs.get("scale", None)

        head_dim = q.meta["val"].shape[-1]
        if scale is None:
            scale = head_dim**-0.5

        q = self.q_node
        k = self.k_node
        v = self.v_node

        assert dropout_p == 0.0
        
        # print("CREATING SDPA WITH FOLLOWING Q, K, V", q.meta["val"].shape, k.meta["val"].shape, v.meta["val"].shape)
        q, k, v, attn_mask = P.slot_map([q, k, v, attn_mask])
        out = P.make_or_get_slot(n)
        P.SDPA(q=q, k=k, v=v, out=out, scale=scale, mask=attn_mask, causal=is_causal)
        return out



def _to_mlx_qparams(qdata: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]: 
    # TorchAO uses s * (q - z), with q signed
    # MLX uses S * Q + B, with Q unsigned
    # s * (q - z) 
    #  = s ((q + offset) - (z + offset))
    #  = s Q + B,
    # where Q = q + offset, B = -s * (z + offset)

    assert qdata.dtype == torch.int8
    offset = 2**(bits - 1)
    Q = qdata.to(torch.int32) + offset
    
    # Pack data tightly into uint32
    assert 32 % bits == 0
    vals_per_uint32 = 32 // bits 
    assert qdata.shape[1] % vals_per_uint32 == 0

    Q = Q.reshape(-1, vals_per_uint32)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)

    # Convert to int64 for shift/packing
    Q = Q.to(torch.int64)
    Q = (Q << shifts).sum(dim=-1)
    Q = Q.to(torch.uint32)
    Q = Q.reshape(qdata.shape[0], -1)
   
    B = -scale * (zero_point.to(scale.dtype) + offset)
    return Q, B

def _parse_dequant_node(node: Node) -> Optional[Tuple[Node, Node, Node, int, int, Optional[torch.dtype]]]:
    qdata, block_size, scale, zero_point, dtype, qmin, qmax  = node.args[0:7]
    out_dtype = node.kwargs.get("output_dtype", None)
    if dtype != torch.int8:
        return None
    if len(block_size) != 2 or block_size[0] != 1 or block_size[1] not in [32, 64, 128]:
        return None
    group_size = block_size[1]
    if (qmin == -8 and qmax == 7):
        bits = 4
    elif (qmin == -128 and qmax == 127):
        bits = 8
    else:
        return None
    return qdata, scale, zero_point, group_size, bits, out_dtype   

@REGISTRY.register_pattern(name="QUANTIZED_LINEAR")
class QuantizedLinearHandler(PatternHandler):
    def __init__(self, head, body, qdata, scale, zero_point, group_size, bits, out_dtype):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.bits = bits
        self.out_dtype = out_dtype
    
    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[QuantizedLinearHandler]:
        _op_namespace = torch.ops.aten

        linear_node = head
        if linear_node.target != _op_namespace.linear.default:
            return None
        
        x, w = linear_node.args[0:2]
        dequant_node = w
        if dequant_node.target != torch.ops.torchao.dequantize_affine.default:
            return None
        
        if dequant_node.users != {linear_node: None}:
            return None

        parsed = _parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype = parsed
        out_dtype = x.meta["val"].dtype if out_dtype is None else out_dtype 

        head = linear_node
        body = [dequant_node]
        return QuantizedLinearHandler(head, body, qdata=qdata, scale=scale, zero_point=zero_point, group_size=group_size, bits=bits, out_dtype=out_dtype)
    
    def __call__(self, P: ProgramBuilder, n: Node):
        assert n == self.head

        x, w = n.args[0:2]
        b = n.args[2] if len(n.args) > 2 else None

        qdata_target, qdata = P.get_placeholder_target_and_tensor(self.qdata)
        zero_point_target, zero_point = P.get_placeholder_target_and_tensor(self.zero_point)
        _, scale = P.get_placeholder_target_and_tensor(self.scale)

        Q, B = _to_mlx_qparams(qdata, scale, zero_point, self.bits)
        out_dtype = str(_TORCH_DTYPE_TO_DTYPEID[self.out_dtype])

        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
        biases = P.make_or_get_constant(f"{zero_point_target}_to_biases", B)

        x, scale, b = P.slot_map([x, self.scale, b])
        out = P.make_or_get_slot(n)
        P.QUANTIZED_LINEAR(x=x, w=w, scales=scale, out=out, biases=biases, group_size=self.group_size, bits=self.bits, mode="affine", out_dtype=out_dtype, bias=b)
        
        return out


@REGISTRY.register_pattern(name="QUANTIZED_EMBEDDING")
class QuantizedEmbeddingHandler(PatternHandler):
    def __init__(self, head, body, qdata, scale, zero_point, group_size, bits, out_dtype):
        super().__init__(head, body)
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.group_size = group_size
        self.bits = bits
        self.out_dtype = out_dtype
    
    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[QuantizedLinearHandler]:
        _op_namespace = torch.ops.aten

        embedding_node = head
        if embedding_node.target !=_op_namespace.embedding.default:
            return None
        
        w, x = embedding_node.args[0:2]

        dequant_node = w
        if dequant_node.target != torch.ops.torchao.dequantize_affine.default:
            return None
        if dequant_node.users != {embedding_node: None}:
            return None

        parsed = _parse_dequant_node(dequant_node)
        if parsed is None:
            return None
        qdata, scale, zero_point, group_size, bits, out_dtype = parsed
        out_dtype = scale.meta["val"].dtype if out_dtype is None else out_dtype 
        head = embedding_node
        body = [dequant_node]
        return QuantizedEmbeddingHandler(head, body, qdata=qdata, scale=scale, zero_point=zero_point, group_size=group_size, bits=bits, out_dtype=out_dtype)
    
    def __call__(self, P: ProgramBuilder, n: Node):
        assert n == self.head
        w, x = n.args[0:2]

        qdata_target, qdata = P.get_placeholder_target_and_tensor(self.qdata)
        zero_point_target, zero_point = P.get_placeholder_target_and_tensor(self.zero_point)
        _, scale = P.get_placeholder_target_and_tensor(self.scale)

        Q, B = _to_mlx_qparams(qdata, scale, zero_point, self.bits)
        out_dtype = str(_TORCH_DTYPE_TO_DTYPEID[self.out_dtype])

        w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
        biases = P.make_or_get_constant(f"{zero_point_target}_to_biases", B)
        x, scale = P.slot_map([x, self.scale])
        out = P.make_or_get_slot(n)
        P.QUANTIZED_GATHER(table_q=w, scales=scale, ids=x, out=out, biases=biases, group_size=self.group_size, bits=self.bits, mode="affine", out_dtype=out_dtype)
        return out
