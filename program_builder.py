# program_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

# Generated mixin (from your codegen)
from generated_ops_mixin import OpsMixin

# Schema utilities (names mirror your ops.hpp)
import ops_schema as S  # must provide READS_WRITES, DTypeId

# -----------------------
# Public slot structures
# -----------------------

class SlotKind:
    Tensor = "Tensor"
    Value = "Value"  # maps to Program::Value (Vid<T>)

@dataclass(frozen=True)
class Slot:
    role: str                 # "const" | "input" | "output" | "mutable_buffer" | "temp"
    kind: str = SlotKind.Tensor
    name: Optional[str] = None
    # For Value slots, declare the underlying Value type ("i32" | "f32" | "bool")
    vtype: Optional[str] = None  # None for tensors

# -----------------------
# ProgramBuilder
# -----------------------

class ProgramBuilder(OpsMixin):
    """
    ProgramBuilder collects slots and instructions using the OpsMixin.
    It keeps Slot handles in payloads; at to_dict() time, it assigns:

      - Tensor IDs (Tid): constants first, then non-constant tensors (unique)
      - Value IDs (Vid<T>): all non-constant scalar slots (unique)

    It also emits explicit I/O maps whose entries are SlotVariants:
      {"domain": "tensor", "idx": ...}
      or {"domain": "value", "idx": ..., "vtype": "i32|f32|bool"}
    """

    # ---- creation ----
    def __init__(self, name: str = ""):
        self.name = name

        # Slots by role (preserve creation order)
        self._consts:  List[Slot] = []
        self._inputs:  List[Slot] = []
        self._outputs: List[Slot] = []
        self._mutable_buffers: List[Slot] = []
        self._temps:   List[Slot] = []

        # Name → Slot (optional)
        self._name_map: Dict[str, Slot] = {}

        # Emitted instructions (op + raw payload with Slot handles)
        self._instrs: List[Dict[str, Any]] = []

        # Liveness bookkeeping for temps (tensor temps only)
        self._temp_first_write: Dict[Slot, int] = {}
        self._temp_last_use:    Dict[Slot, int] = {}

        # Scalar defaults for runtime seeding of Vid<int> etc.
        self.value_input_defaults: Dict[str, Any] = {}

    # ---- slot APIs ----
    def _add_slot(self, role: str, name: Optional[str], kind: str, vtype: Optional[str] = None) -> Slot:
        if role == "const" and kind != SlotKind.Tensor:
            raise ValueError("Constants must be Tensor kind (no value constants).")
        if kind == SlotKind.Value and vtype not in ("i32", "f32", "bool"):
            raise ValueError(f"Value slot requires vtype in ('i32','f32','bool'), got {vtype}")
        s = Slot(role=role, name=name, kind=kind, vtype=vtype)
        if name:
            if name in self._name_map:
                # Point the name to the *first* slot; disallow silent overwrite.
                raise ValueError(f"Duplicate slot name: {name}")
            self._name_map[name] = s
        if role == "const":
            self._consts.append(s)
        elif role == "input":
            self._inputs.append(s)
        elif role == "output":
            self._outputs.append(s)
        elif role == "mutable_buffer":
            self._mutable_buffers.append(s)
        elif role == "temp":
            self._temps.append(s)
        else:
            raise ValueError(f"Unknown role: {role}")
        return s

    def constant(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor) -> Slot:
        return self._add_slot("const", name, kind)

    def input(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor, vtype: Optional[str] = None) -> Slot:
        return self._add_slot("input", name, kind, vtype=vtype)

    def output(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor, vtype: Optional[str] = None) -> Slot:
        return self._add_slot("output", name, kind, vtype=vtype)

    def mutable_buffer(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor, vtype: Optional[str] = None) -> Slot:
        return self._add_slot("mutable_buffer", name, kind, vtype=vtype)

    # Back-compat alias
    def mbuf(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor, vtype: Optional[str] = None) -> Slot:
        return self.mutable_buffer(name=name, kind=kind, vtype=vtype)

    def temp(self, name: Optional[str] = None, *, kind: str = SlotKind.Tensor, vtype: Optional[str] = None) -> Slot:
        return self._add_slot("temp", name, kind, vtype=vtype)

    # Convenience for scalar runtime inputs (Vid<int32_t>)
    def scalar_input_i32(self, name: str, default: Optional[int] = None) -> Slot:
        s = self.input(name=name, kind=SlotKind.Value, vtype="i32")
        if default is not None:
            self.value_input_defaults[name] = int(default)
        return s

    # Expose an existing tensor Slot as an output (without allocating a new tensor)
    def expose_output(self, slot: Slot, name: Optional[str] = None) -> None:
        if slot.kind != SlotKind.Tensor:
            raise TypeError("Only tensor slots can be exposed as outputs")
        self._outputs.append(slot)
        if name and name not in self._name_map:
            self._name_map[name] = slot

    # ---- mixin hooks ----
    def _coerce_payload(self, op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            if k == "self":
                continue
            if isinstance(v, S.DTypeId):
                out[k] = v.name  # e.g., "f16"
            else:
                out[k] = v
        return out

    def _emit(self, op: str, **payload) -> None:
        idx = len(self._instrs)
        self._instrs.append({"op": op, **payload})

        reads, writes = S.READS_WRITES.get(op, ((), ()))
        # writers
        for fname in writes:
            s = payload.get(fname)
            if isinstance(s, Slot) and s.role == "temp" and s.kind == SlotKind.Tensor:
                self._temp_first_write.setdefault(s, idx)
                self._temp_last_use[s] = idx
        # reads
        for fname in reads:
            s = payload.get(fname)
            if isinstance(s, Slot) and s.role == SlotKind.Tensor:
                # only temps participate in liveness
                if s.role == "temp":
                    self._temp_last_use[s] = idx

    # ---- finalize / JSON manifest ----

    def _iter_all_slots_ordered(self) -> Iterable[Slot]:
        yield from self._consts
        yield from self._inputs
        yield from self._outputs
        yield from self._mutable_buffers
        yield from self._temps

    def _assign_ids(self) -> Tuple[Dict[Slot, int], Dict[Slot, int]]:
        """
        Assign dense IDs once per unique Slot.
          - TID: constants (tensors) first, then unique non-constant tensors (inputs, outputs, mbufs, temps)
          - VID: all unique non-constant Value slots
        Returns (tensor_id_map, value_id_map)
        """
        tid_map: Dict[Slot, int] = {}
        vid_map: Dict[Slot, int] = {}

        cursor_tid = 0
        # constants (tensors only)
        for s in self._consts:
            if s.kind == SlotKind.Tensor and s not in tid_map:
                tid_map[s] = cursor_tid; cursor_tid += 1

        def add_tensor_block(block: List[Slot]):
            nonlocal cursor_tid
            for s in block:
                if s.kind == SlotKind.Tensor and s not in tid_map:
                    tid_map[s] = cursor_tid; cursor_tid += 1

        add_tensor_block(self._inputs)
        add_tensor_block(self._outputs)
        add_tensor_block(self._mutable_buffers)
        add_tensor_block(self._temps)

        cursor_vid = 0
        def add_value_block(block: List[Slot]):
            nonlocal cursor_vid
            for s in block:
                if s.kind == SlotKind.Value and s not in vid_map:
                    vid_map[s] = cursor_vid; cursor_vid += 1

        add_value_block(self._inputs)
        add_value_block(self._outputs)
        add_value_block(self._mutable_buffers)
        add_value_block(self._temps)

        return tid_map, vid_map

    @staticmethod
    def _encode_slot_variant(s: Slot, tid_map: Dict[Slot, int], vid_map: Dict[Slot, int]) -> Dict[str, Any]:
        if s.kind == SlotKind.Tensor:
            return {"domain": "tensor", "idx": int(tid_map[s])}
        else:
            return {"domain": "value", "idx": int(vid_map[s]), "vtype": s.vtype}

    def _vid_of(self, tid_map: Dict[Slot, int], vid_map: Dict[Slot, int], maybe_slot) -> Any:
        if isinstance(maybe_slot, Slot):
            if maybe_slot.kind == SlotKind.Tensor:
                return {"tid": int(tid_map[maybe_slot])}
            else:
                return {"vid": int(vid_map[maybe_slot]), "vtype": maybe_slot.vtype}
        return maybe_slot

    def _rewrite_instrs(self, tid_map: Dict[Slot, int], vid_map: Dict[Slot, int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for instr in self._instrs:
            op = instr["op"]
            payload = {k: v for k, v in instr.items() if k != "op"}
            rew = {k: self._vid_of(tid_map, vid_map, v) for k, v in payload.items()}
            out.append({"op": op, **rew})
        return out

    def _make_tensor_meta(self) -> List[Optional[Dict[str, Any]]]:
        """Optional tensor metadata vector aligned to TID space (size == max_tid+1)."""
        tid_map, _ = self._assign_ids()
        num_tensors = (max(tid_map.values()) + 1) if tid_map else 0
        meta: List[Optional[Dict[str, Any]]] = [None] * num_tensors
        for s, tid in tid_map.items():
            meta[tid] = {"name": s.name if s.name else None}
        return meta

    def _make_name_to_slotvariant(self, tid_map: Dict[Slot, int], vid_map: Dict[Slot, int]) -> Dict[str, Dict[str, Any]]:
        nm: Dict[str, Dict[str, Any]] = {}
        for name, slot in self._name_map.items():
            # only encode if the slot actually has an id in its domain
            if slot.kind == SlotKind.Tensor and slot in tid_map:
                nm[name] = self._encode_slot_variant(slot, tid_map, vid_map)
            elif slot.kind == SlotKind.Value and slot in vid_map:
                nm[name] = self._encode_slot_variant(slot, tid_map, vid_map)
        return nm

    def _temp_liveness(self) -> List[Optional[Tuple[int, int]]]:
        out: List[Optional[Tuple[int, int]]] = []
        for s in self._temps:
            if s.kind != SlotKind.Tensor:
                out.append(None); continue
            fw = self._temp_first_write.get(s, None)
            lu = self._temp_last_use.get(s, fw)
            if fw is None:
                out.append(None)
            else:
                out.append((fw, lu if lu is not None else fw))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """
        Finalize and produce a JSON-serializable manifest (no tensor data).
        """
        # Assign IDs
        tid_map, vid_map = self._assign_ids()

        # Counts must reflect UNIQUE slots in each domain
        num_constant_tensors = sum(1 for s in self._consts if s.kind == SlotKind.Tensor)
        num_total_tensors = (max(tid_map.values()) + 1) if tid_map else 0
        num_non_constant_tensors = num_total_tensors - num_constant_tensors

        num_total_values = (max(vid_map.values()) + 1) if vid_map else 0
        num_non_constant_values = num_total_values  # all values are non-constant in this design

        # Rewrite instructions
        instrs = self._rewrite_instrs(tid_map, vid_map)

        # I/O maps (positional order, may contain duplicates → same idx)
        input_map = [self._encode_slot_variant(s, tid_map, vid_map) for s in self._inputs]
        output_map = [self._encode_slot_variant(s, tid_map, vid_map) for s in self._outputs]
        mutable_buffer_map = [self._encode_slot_variant(s, tid_map, vid_map) for s in self._mutable_buffers]

        manifest: Dict[str, Any] = {
            "program_name": self.name,
            "num_constant_tensors": num_constant_tensors,
            "num_non_constant_tensors": num_non_constant_tensors,
            "num_non_constant_values": num_non_constant_values,
            "tensor_meta": self._make_tensor_meta(),          # aligned to TID space
            "instructions": instrs,
            "name_to_slot": self._make_name_to_slotvariant(tid_map, vid_map),
            "value_input_defaults": dict(self.value_input_defaults),
            "input_map": input_map,
            "output_map": output_map,
            "mutable_buffer_map": mutable_buffer_map,
            "temp_lifetimes": self._temp_liveness(),
        }
        return manifest

    # ---- draft / debugging helpers ----

    def _draftify_value(self, v: Any) -> Any:
        if isinstance(v, Slot):
            base = {"role": v.role, "kind": v.kind, "name": v.name}
            if v.kind == SlotKind.Value:
                base["vtype"] = v.vtype
            return base
        if isinstance(v, S.DTypeId):
            return v.name
        return v

    def draft_snapshot(self) -> Dict[str, Any]:
        slots = [
            ({"role": s.role, "kind": s.kind, "name": s.name} if s.kind == SlotKind.Tensor
             else {"role": s.role, "kind": s.kind, "name": s.name, "vtype": s.vtype})
            for s in self._iter_all_slots_ordered()
        ]
        instrs = []
        for instr in self._instrs:
            op = instr["op"]
            payload = {k: v for k, v in instr.items() if k != "op"}
            pretty = {k: self._draftify_value(v) for k, v in payload.items()}
            instrs.append({"op": op, **pretty})
        return {
            "program_name": self.name,
            "slots": slots,
            "instructions": instrs,
            "value_input_defaults": dict(self.value_input_defaults),
        }
