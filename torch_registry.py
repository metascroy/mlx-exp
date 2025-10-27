# torch_registry.py — robust view materialization + output exposure
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as fx
from torch.fx.node import Node

from program_builder import ProgramBuilder, Slot, SlotKind
from ops_schema import DTypeId

Handler = Callable[["LowerCtx", Node], Optional[Slot]]
CapPredicate = Callable[[Node], bool]
DEFERRED = object()

class TorchOpRegistry:
    def __init__(self):
        self._by_target: Dict[Union[str, Callable], "Entry"] = {}

    def register(self, target: Union[str, Callable, list, tuple], *, capability: Optional[CapPredicate] = None, name: Optional[str] = None):
        def deco(fn: Handler):
            targets = target if isinstance(target, (list, tuple)) else [target]
            for t in targets:
                self._by_target[t] = Entry(target=t, handler=fn, capability=capability, name=name or getattr(fn, "__name__", str(t)))
            return fn
        return deco

    def _normalize_target(self, t: Any) -> List[Any]:
        out = []
        if isinstance(t, str):
            out.append(t)
            if "." in t:
                out.append(t.rsplit(".", 1)[0])
            return out
        name = getattr(t, "name", None)
        if isinstance(name, str):
            out += [name, name.rsplit(".", 1)[0]] if "." in name else [name]
        if callable(t):
            nm = getattr(t, "__name__", None)
            qn = getattr(t, "__qualname__", None)
            if isinstance(nm, str): out.append(nm)
            if isinstance(qn, str): out.append(qn)
            out.append(str(t))
        return out

    def get(self, node: Node) -> Optional["Entry"]:
        t = node.target
        if t in self._by_target:
            return self._by_target[t]
        for key in self._normalize_target(t):
            if key in self._by_target:
                return self._by_target[key]
        return None

@dataclass
class Entry:
    target: Union[str, Callable]
    handler: Handler
    capability: Optional[CapPredicate] = None
    name: Optional[str] = None

REGISTRY = TorchOpRegistry()

@dataclass
class LowerCtx:
    P: ProgramBuilder
    gm: fx.GraphModule
    draft: bool = False
    annotate_fx: bool = True
    values: Dict[Node, Slot] = None
    pending_views: Dict[Node, Tuple[Node, int, int, int]] = None
    ops_namespace = torch._ops.ops.aten

    def __post_init__(self):
        if self.values is None: self.values = {}
        if self.pending_views is None: self.pending_views = {}

    # ---- accessors ----
    def as_tensor(self, node: Node) -> Slot:
        s = self.values.get(node)
        if s is None or s.kind != SlotKind.Tensor:
            raise KeyError(f"Tensor slot missing for node {node}")
        return s

    def scalar_input_i32(self, name: str, default: Optional[int] = None) -> Slot:
        return self.P.scalar_input_i32(name, default=default)

    def const_param(self, t: torch.Tensor, name: Optional[str] = None) -> Slot:
        return self.P.constant(name=name, kind=SlotKind.Tensor)

    def set_out(self, node: Node, slot: Slot):
        self.values[node] = slot
        if self.annotate_fx:
            node.meta.setdefault("lower", {})
            node.meta["lower"]["slot"] = slot.name or f"{slot.role}:{slot.kind}"

    def mark_support(self, node: Node, ok: bool, reason: Optional[str] = None, handler: Optional[str] = None):
        if self.annotate_fx:
            m = node.meta.setdefault("support", {})
            m["supported"] = bool(ok)
            if handler: m["handler"] = handler
            if reason:  m["reason"] = reason

# ------------------------------
# Lowering driver
# ------------------------------

def lower_graph(gm: fx.GraphModule, *, draft: bool = False, name: str = "program", annotate_fx: bool = True):
    P = ProgramBuilder(name)
    ctx = LowerCtx(P=P, gm=gm, draft=draft, annotate_fx=annotate_fx)

    try:
        # Prime placeholders as inputs
        for n in gm.graph.nodes:
            if n.op != "placeholder":
                continue
            s = P.input(name=n.name, kind=SlotKind.Tensor)
            ctx.set_out(n, s)
            ctx.mark_support(n, ok=True, handler="placeholder")

        # Walk compute nodes
        for n in gm.graph.nodes:
            if n.op in ("placeholder", "output"): continue
            entry = REGISTRY.get(n)
            if entry is None:
                msg = f"no handler for target={n.target}"
                if draft: ctx.mark_support(n, ok=False, reason=msg); continue
                raise NotImplementedError(msg)

            if entry.capability and not entry.capability(n):
                msg = f"{entry.name}: capability predicate failed"
                if draft: ctx.mark_support(n, ok=False, reason=msg, handler=entry.name); continue
                raise NotImplementedError(msg)

            try:
                out_slot = entry.handler(ctx, n)
                ctx.mark_support(n, ok=True, handler=entry.name)
                if out_slot not in (None, DEFERRED):
                    ctx.set_out(n, out_slot)
            except Exception as e:
                msg = f"{entry.name} failed for {n.target}: {e}"
                if draft: ctx.mark_support(n, ok=False, reason=msg, handler=entry.name); continue
                raise RuntimeError(msg) from e

        # Collect & expose outputs (realize pending views if needed)
        out_nodes: List[Node] = []
        for n in gm.graph.nodes:
            if n.op == "output":
                ret = n.args[0]
                out_nodes = list(ret) if isinstance(ret, (list, tuple)) else [ret]
                break

        for i, on in enumerate(out_nodes):
            if not isinstance(on, Node):
                raise NotImplementedError("Literal outputs not supported")
            s_in = _realize_view_if_needed(ctx, on) if on in ctx.pending_views else ctx.as_tensor(on)
            ctx.P.expose_output(s_in, name=f"out{i}")

        if draft:
            return None, gm, {"draft_snapshot": P.draft_snapshot(), "support": _support_summary(gm)}

        manifest = P.to_dict()
        manifest["support"] = _support_summary(gm)
        return P, gm, manifest

    finally:
        ctx.pending_views.clear()

def _support_summary(gm: fx.GraphModule) -> Dict[str, Any]:
    total, ok = 0, 0
    rows = []
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output"): continue
        total += 1
        s = n.meta.get("support", {})
        rows.append({
            "name": n.name,
            "target": str(n.target),
            "supported": bool(s.get("supported", False)),
            "handler": s.get("handler"),
            "reason": s.get("reason"),
        })
        if s.get("supported", False): ok += 1
    return {"total": total, "supported": ok, "nodes": rows}

# ------------------------------
# Helpers
# ------------------------------

def _get_tensor_slot(ctx: LowerCtx, x: Any, name_hint: Optional[str] = None) -> Slot:
    if isinstance(x, Node):
        if x in ctx.values:
            return ctx.values[x]
        if x in ctx.pending_views:
            return _realize_view_if_needed(ctx, x)
        raise KeyError(f"Tensor value for node {x} not set and not a pending view")
    if isinstance(x, torch.Tensor):
        return ctx.const_param(x, name=name_hint)
    raise TypeError(f"Expected Node or Tensor, got {type(x)}")

def _int_from_arg(a: Any) -> int:
    if isinstance(a, int): return a
    try: return int(a)
    except Exception: pass
    if isinstance(a, torch.Tensor) and a.ndim == 0 and a.dtype in (torch.int32, torch.int64):
        return int(a.item())
    raise TypeError(f"Cannot extract int from {type(a)}")

# ------------------------------
# View materialization
# ------------------------------

def _extract_view(ops_namespace, n: Node) -> Optional[tuple[Node, int, int, int]]:
    if n.target == ops_namespace.narrow.default:
        base, dim, start, length = n.args
        return base, int(dim), int(start), int(length)
    if n.target == ops_namespace.slice.Tensor:
        base, dim, start, end, step = n.args
        if step not in (1, None):
            raise NotImplementedError("slice step != 1")
        length = int(end) - int(start) if end is not None else -1
        return base, int(dim), int(start), int(length)
    return None

def _realize_view_if_needed(ctx: LowerCtx, node: Node) -> Slot:
    if node in ctx.values:
        return ctx.values[node]
    if node in ctx.pending_views:
        base, dim, start, length = ctx.pending_views.pop(node)
        baseS = ctx.as_tensor(base)
        axisS = ctx.scalar_input_i32(f"{node.name}_axis", default=dim)
        startS = ctx.scalar_input_i32(f"{node.name}_start", default=start)
        lengthS = ctx.scalar_input_i32(f"{node.name}_len", default=length)
        outS = ctx.P.temp(node.name)
        ctx.P.SLICE(x=baseS, out=outS, axis=axisS, start=startS, length=lengthS)
        ctx.set_out(node, outS)
        return outS
    raise KeyError(f"No slot or pending view for node {node}")

# ------------------------------
# Handlers
# ------------------------------

@REGISTRY.register(target="aten.linear.default")
def handle_linear(ctx: LowerCtx, n: Node) -> Slot:
    x, w, b = n.args
    xS = _get_tensor_slot(ctx, x, "x")
    wS = _get_tensor_slot(ctx, w, "w")
    outS = ctx.P.temp(name=n.name)
    if b is None:
        ctx.P.LINEAR(x=xS, weight=wS, out=outS, bias=None)
    else:
        bS = _get_tensor_slot(ctx, b, "bias")
        ctx.P.LINEAR(x=xS, weight=wS, out=outS, bias=bS)
    return outS

@REGISTRY.register(target="aten.zeros.default")
def handle_zeros(ctx: LowerCtx, n: Node) -> Slot:
    shape_arg = n.args[0]
    if isinstance(shape_arg, (list, tuple)):
        shape = [int(v) for v in shape_arg]
    elif isinstance(shape_arg, torch.Tensor):
        shape = [int(x) for x in shape_arg.tolist()]
    elif isinstance(shape_arg, Node):
        raise NotImplementedError("dynamic shape node not yet supported")
    else:
        raise TypeError(f"zeros: unsupported shape arg type {type(shape_arg)}")

    # dtype may be None → use torch's default dtype policy
    dtype_kw = n.kwargs.get("dtype", None)
    if dtype_kw is None:
        cur = torch.get_default_dtype()
        if cur == torch.float32:     dtype_kw = torch.float32
        elif cur == torch.float64:   dtype_kw = torch.float32  # snap to f32
        else:                        dtype_kw = torch.float16  # conservative fallback

    dtype_map = {
        torch.float32: DTypeId.f32,
        torch.float16: DTypeId.f16,
        torch.bfloat16: DTypeId.bf16,
        torch.int32:   DTypeId.i32,
        torch.int64:   DTypeId.i64,
        torch.uint8:   DTypeId.u8,
        torch.bool:    DTypeId.bool if hasattr(DTypeId, "bool") else DTypeId.u8,
    }
    dtype = dtype_map.get(dtype_kw, DTypeId.f16)

    outS = ctx.P.temp(name=n.name)
    ctx.P.ZEROS(out=outS, shape=shape, dtype=dtype)
    return outS

@REGISTRY.register(target="aten.add.Tensor")
def handle_add(ctx: LowerCtx, n: Node) -> Slot:
    a, b = n.args[:2]
    aS = _get_tensor_slot(ctx, a, "a")
    bS = _get_tensor_slot(ctx, b, "b")
    outS = ctx.P.temp(name=n.name)
    ctx.P.ADD(a=aS, b=bS, out=outS)
    return outS

@REGISTRY.register(target="aten.mul.Tensor")
def handle_mul(ctx: LowerCtx, n: Node) -> Slot:
    a, b = n.args[:2]
    aS = _get_tensor_slot(ctx, a, "a")
    bS = _get_tensor_slot(ctx, b, "b")
    outS = ctx.P.temp(name=n.name)
    ctx.P.MUL(a=aS, b=bS, out=outS)
    return outS

@REGISTRY.register(target="aten.relu.default")
def handle_relu_via_silu(ctx: LowerCtx, n: Node) -> Slot:
    x = n.args[0]
    xS = _get_tensor_slot(ctx, x, "x")
    outS = ctx.P.temp(name=n.name)
    ctx.P.SILU(x=xS, out=outS)
    return outS

@REGISTRY.register(target="aten.reshape.default")
def handle_reshape(ctx: LowerCtx, n: Node) -> Slot:
    x, shape = n.args
    xS = _get_tensor_slot(ctx, x, "x")
    outS = ctx.P.temp(name=n.name)
    if isinstance(shape, (list, tuple)):
        shape_list = list(shape)
    elif isinstance(shape, torch.Tensor):
        shape_list = [int(v) for v in shape.tolist()]
    else:
        try:
            shape_list = [int(v) for v in shape]
        except Exception as e:
            raise TypeError(f"reshape: unsupported shape type {type(shape)}") from e
    ctx.P.RESHAPE(x=xS, out=outS, shape=[int(v) for v in shape_list])
    return outS

@REGISTRY.register(target="aten.transpose.int")
def handle_transpose(ctx: LowerCtx, n: Node) -> Slot:
    x, dim0, dim1 = n.args
    xS = _get_tensor_slot(ctx, x)
    outS = ctx.P.temp(name=n.name)

    # Safer rank discovery from meta
    r = None
    tm = x.meta.get("tensor_meta")
    if tm is not None:
        if hasattr(tm, "shape"):
            try: r = len(tm.shape)
            except Exception: r = None
        elif isinstance(tm, dict) and "shape" in tm and tm["shape"] is not None:
            r = len(tm["shape"])
        elif hasattr(tm, "dim"):
            try: r = int(tm.dim())
            except Exception: r = None

    if r is None:
        perm = [int(dim0), int(dim1)]
    else:
        perm = list(range(r))
        d0, d1 = int(dim0), int(dim1)
        perm[d0], perm[d1] = perm[d1], perm[d0]

    ctx.P.TRANSPOSE(x=xS, out=outS, perm=perm)
    return outS

@REGISTRY.register(target="aten.contiguous.default")
def handle_contiguous(ctx: LowerCtx, n: Node) -> Slot:
    x = n.args[0]
    xS = _get_tensor_slot(ctx, x)
    outS = ctx.P.temp(name=n.name)
    ctx.P.CONTIGUOUS(x=xS, out=outS)
    return outS

@REGISTRY.register(target="aten.cat.default")
def handle_concat(ctx: LowerCtx, n: Node) -> Slot:
    tensors, dim = n.args
    assert len(tensors) == 2, "Current CONCAT handler supports a pair; extend for N-way"
    aS = _get_tensor_slot(ctx, tensors[0], "a")
    bS = _get_tensor_slot(ctx, tensors[1], "b")
    axis = _int_from_arg(dim)
    outS = ctx.P.temp(name=n.name)
    ctx.P.CONCAT(a=aS, b=bS, out=outS, axis=axis)
    return outS

# ---- deferred views ----

@REGISTRY.register(target="aten.narrow.default")
def handle_narrow_deferred(ctx: LowerCtx, n: Node) -> Optional[Slot]:
    info = _extract_view(ctx.ops_namespace, n)
    if info is None: raise TypeError("invalid narrow args")
    ctx.pending_views[n] = info
    return DEFERRED

@REGISTRY.register(target="aten.slice.Tensor")
def handle_slice_deferred(ctx: LowerCtx, n: Node) -> Optional[Slot]:
    info = _extract_view(ctx.ops_namespace, n)
    if info is None: raise TypeError("invalid slice args")
    ctx.pending_views[n] = info
    return DEFERRED

@REGISTRY.register(target="aten.copy_.default")
def handle_copy_into_view(ctx: LowerCtx, n: Node) -> Optional[Slot]:
    dst: Node = n.args[0]
    src: Node = n.args[1]

    if dst in ctx.pending_views:
        base, dim, start, length = ctx.pending_views.pop(dst)
        baseS = _realize_view_if_needed(ctx, base)
        srcS = _realize_view_if_needed(ctx, src) if isinstance(src, Node) else _get_tensor_slot(ctx, src)
        axisS = ctx.scalar_input_i32(f"{n.name}_axis", default=dim)
        startS = ctx.scalar_input_i32(f"{n.name}_start", default=start)
        lengthS = ctx.scalar_input_i32(f"{n.name}_len", default=length)
        ctx.P.SLICE_UPDATE(dst=baseS, update=srcS, axis=axisS, start=startS, length=lengthS)
        ctx.set_out(n, baseS)
        return baseS

    # No pending view: act as no-op; return destination tensor
    dstS = _realize_view_if_needed(ctx, dst) if isinstance(dst, Node) else _get_tensor_slot(ctx, dst)
    ctx.set_out(n, dstS)
    return dstS
