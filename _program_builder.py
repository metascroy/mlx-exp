# program_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from collections import defaultdict
import tqdm
import uuid

# Generated mixin (from your codegen)
from generated_ops_mixin import OpsMixin
import torch

# Schema utilities (names mirror your ops.hpp)
import ops_schema as S  # must provide READS_WRITES, DTypeId
from torch.fx.node import Node
from torch.utils import _pytree as pytree

from dataclasses import dataclass, field
from typing import Optional, List, DefaultDict, Union
from collections import defaultdict
import traceback
import json

# -----------------------
# ProgramBuilder
# -----------------------

Handler = Callable[["ProgramBuilder", Node], Optional["Slot"]]

class PatternHandler:
    def __init__(self, head: Node, body: List[Node]) -> None:
        self.head: Node = head
        self.body: List[Node] = body
    
    @classmethod
    def deferred_handler(cls, P: "ProgramBuilder", n: Node) -> None:
        pass

    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node) -> Optional[PatternHandler]:
        raise NotImplementedError
    
    def __call__(self, P: "ProgramBuilder", n: Node) -> None:
        raise NotImplementedError

    def set_handlers(self, P: "ProgramBuilder"):
        assert P.node_info[self.head].handler is None
        for n in self.body:
            assert P.node_info[n].handler is None

        P.node_info[self.head].handler = self
        for n in self.body:
            P.node_info[n].handler = PatternHandler.deferred_handler

class TorchOpRegistry:
    def __init__(self):
        self._by_target: Dict[Union[str, Callable], "Entry"] = {}
        self._patterns: Dict[str, Type[PatternHandler]] = {}
        self.ops_namespace = torch._ops.ops.aten

    def register(self, target: Union[str, Callable, list, tuple]):
        def deco(fn: Handler):
            targets = target if isinstance(target, (list, tuple)) else [target]
            for t in targets:
                if t in self._by_target:
                    raise ValueError(f"Target {t} already registered") 
                self._by_target[t] = Entry(target=t, handler=fn)
            return fn
        return deco

    def get(self, node: Node) -> Optional["Entry"]:
        t = node.target
        if t in self._by_target:
            return self._by_target[t]
        return None
    
    def register_pattern(self, name: str):
        """
        Class decorator for PatternHandler subclasses.
        """
        def deco(cls: Type[PatternHandler]):
            if not issubclass(cls, PatternHandler):
                raise TypeError("register_pattern must decorate a PatternHandler subclass")
            if name in self._patterns:
                raise ValueError(f"Pattern '{name}' already registered")
            self._patterns[name] = cls
            return cls
        return deco

    def get_pattern_cls(self, name: str) -> Optional[Type[PatternHandler]]:
        return self._patterns.get(name)
    
    def patterns(self) -> Iterable[str]:
        return self._patterns.keys()

@dataclass
class Entry:
    target: Union[str, Callable]
    handler: Handler

REGISTRY = TorchOpRegistry()


from enum import Enum, auto

class IdType(Enum):
    Tensor = auto()
    SymInt = auto()
    SymBool = auto()

class IdSpace(Enum):
    Constant = auto()
    Input = auto()
    Output = auto()
    MutableBuffer = auto()
    Temp = auto()

@dataclass(frozen=True)
class Slot:
    id_type: IdType
    id_space: IdSpace
    idx: Optional[int] = None


class IdManager:
    def __init__(self):
        self.free: list[int] = []
        self.next_new_id = 0

    def get_id(self):
        return self.free.pop() if self.free else self._bump()

    def _bump(self):
        idx = self.next_new_id
        self.next_new_id += 1
        return idx

    def return_id(self, idx):
        if self.free and self.free[-1] == idx:
            return
        self.free.append(idx)
    

class SlotManager:
    def __init__(self):
        self.tid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.vid_managers: Dict[IdSpace, IdManager] = defaultdict(IdManager)
        self.name_to_slot: Dict[str, Slot] = {}
    
    def set_slot(self, node_or_name: Union[Node, str], slot: Slot):
        if isinstance(node_or_name, Node):
            node_or_name = node_or_name.name
        assert node_or_name not in self.name_to_slot
        self.name_to_slot[node_or_name] = slot  
    
    def get_slot(self, node_or_name: Union[Node, str]) -> Optional[Union[Tuple[Slot], Slot]]:
        if isinstance(node_or_name, Node):
            node_or_name = node_or_name.name
        return self.name_to_slot.get(node_or_name, None)
    
    def _val_to_idtype(self, v) -> IdType:
        from torch._subclasses.fake_tensor import FakeTensor
        if isinstance(v, FakeTensor):
            return IdType.Tensor
        elif isinstance(v, torch.SymInt):
            return IdType.SymInt
        elif isinstance(v, torch.SymBool):
            return IdType.SymBool
        else:
            raise NotImplementedError(f"val_to_idtype: {v}")

        
    def is_alive(self, slot: Slot) -> bool:
        if slot.id_type == IdType.Tensor:
            manager = self.tid_managers[slot.id_space]
        else:
            manager = self.vid_managers[slot.id_space]
        idx = slot.idx
        if idx >= manager.next_new_id:
            return False
        if idx in manager.free:
            return False
        return True
    
    def make_constant_slot(self, name: str) -> Slot:
        assert name not in self.name_to_slot
        id_space = IdSpace.Constant
        manager = self.tid_managers[id_space]
        idx = manager.get_id()
        slot = Slot(id_type=IdType.Tensor, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        return slot
    
    def make_tmp_slot(self) -> Tuple[str, Slot]:
        name = f"tmp_{uuid.uuid4().hex}"
        id_space = IdSpace.Temp
        manager = self.tid_managers[id_space]
        idx = manager.get_id()
        slot = Slot(id_type=IdType.Tensor, id_space=id_space, idx=idx)
        self.name_to_slot[name] = slot
        return name, slot

    def make_or_get_slot(self, node, id_space: IdSpace = IdSpace.Temp) -> Union[Slot, Tuple[Slot]]:
        if node.name in self.name_to_slot:
            slot = self.name_to_slot[node.name]
            return slot

        val = node.meta.get("val", None)
        assert val is not None, f"Node {node} has no val"
        if not isinstance(val, (list, tuple)):
            val = (val,)
        
        slots = []
        for v in val:
            id_type = self._val_to_idtype(v)
            if id_type == IdType.Tensor:
                manager = self.tid_managers[id_space]
            else:
                manager = self.vid_managers[id_space]
            idx = manager.get_id()
            slots.append(Slot(id_type=id_type, id_space=id_space, idx=idx))
        slots = tuple(slots)

        if len(slots) == 1:
            slots = slots[0]
        
        self.set_slot(node, slots)
        return slots

@dataclass
class NodeInfo:
    handled: bool = False # Whether we've considered it
    handler: Optional[Union[Handler, PatternHandler]] = None
    supported: bool = False # Whether it should be supported in partitioner
    unsupported_reason: Optional[str] = None
    name: Optional[str] = None
    remaining_reads = 0

class ProgramBuilder(OpsMixin):
    def __init__(self, ep: ExportedProgram):
        self.tid = {}
        self.vid = {}
        self._instrs = []
        self.ep: ExportedProgram = ep
        self.extra_constants = {}
        self.slot_manager = SlotManager()
        self.node_info: DefaultDict[Node, NodeInfo] = defaultdict(NodeInfo)
        self.prog_json: Optional[str] = None
    
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
    
    def get_placeholder_target_and_tensor(self, node: Node) -> Tuple[str, torch.Tensor]:
        assert node.op == "placeholder"
        placeholder_name = node.name
        from torch.export.graph_signature import InputKind
        sig = self.ep.graph_signature
        sd = self.ep.state_dict
        consts = self.ep.constants

        for ispec in sig.input_specs:
            if ispec.arg.name != placeholder_name:
                continue
            target = ispec.target
            if target is None:
                continue
            if target in sd:
                return (target, sd[target])
            if target in consts:
                return (target, consts[target])

        raise KeyError(f"Unable to resolve placeholder {placeholder_name}")

    def args(self, node: Node) -> Tuple[Any, ...]:
        return self.slot_map(node.args)
    
    def kwargs(self, node: Node) -> Dict[str, Any]:
        return self.slot_map(node.kwargs)
    
    def slot_map(self, tree):
        leaves, spec = pytree.tree_flatten(tree)
        new_leaves = []
        for a in leaves:
            if isinstance(a, Node):
                new_leaves.append(self.make_or_get_slot(a))
            else:
                new_leaves.append(a)

        # Verify that no slot is already freed
        for a in new_leaves:
            if isinstance(a, Slot):
                assert self.slot_manager.is_alive(a), f"Slot {a} is not alive; it was either already freed or never created"

        return pytree.tree_unflatten(new_leaves, spec)
    
    def make_or_get_constant(self, name: str, tensor: torch.Tensor) -> Slot:
        """
        Creates an extra constant outside of the ExportedProgram state_dict
        Ops can use this to add constants during build that do not exist in the
        ExportedProgram state_dict, e.g., doing naive packing of quantized ops 
        """
        assert name not in self.ep.state_dict
        assert name not in self.ep.constants

        if name in self.extra_constants:
            assert torch.equal(tensor, self.extra_constants[name])
            slot = self.slot_manager.get_slot(name)
            assert slot is not None
            return slot

        slot = self.slot_manager.make_constant_slot(name)
        self.extra_constants[name] = tensor
        return slot
    
    def make_or_get_slot(self, node, id_space: IdSpace = IdSpace.Temp) -> Slot:
        return self.slot_manager.make_or_get_slot(node, id_space)
    
    def set_slot(self, node, slot: Slot):
        self.slot_manager.set_slot(node, slot)
    
    def _mark_read(self, node: Node):
        assert self.node_info[node].handled, f"Node {node} is not handled"
        assert self.node_info[node].remaining_reads > 0, f"Reading node {node}, but it has no remaining reads"
        self.node_info[node].remaining_reads -= 1

        if self.node_info[node].remaining_reads == 0:
            slot = self.slot_manager.get_slot(node)
            if slot is None:
                return
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                if s.id_space != IdSpace.Temp:
                    continue
                if s.id_type == IdType.Tensor:
                    self.slot_manager.tid_managers[IdSpace.Temp].return_id(s.idx)
                else:
                    self.slot_manager.vid_managers[IdSpace.Temp].return_id(s.idx)

    def _mark_node_handled(self, node: Node, *, handler: Optional[Handler] = None):
        if self.node_info[node].handled:
            return
        self.node_info[node].handled = True
        self.node_info[node].remaining_reads = len(node.users)
        self.node_info[node].handler = handler

        # Do not mark reads on a deferred handler
        # The nodes must be kept alive until the pattern is handled
        if handler == PatternHandler.deferred_handler:
            return

        def mark_read(n: Node):
            flat_args, spec = pytree.tree_flatten((n.args, n.kwargs))
            # The same node can occur multiple time in flat_args, we need to only mark it read once
            seen = set()
            for a in flat_args:
                if isinstance(a, Node):
                    if a not in seen:
                        self._mark_read(a)
                        seen.add(a)

        if isinstance(handler, PatternHandler):
            for n in handler.body:
                mark_read(n)
        mark_read(node)

    def _mark_node_supported(self, node: Node, *, handler: Optional[Handler] = None):
        self.node_info[node].supported = True
        self._mark_node_handled(node, handler=handler)
    
    def _mark_node_unsupported(self, node: Node, reason: str):
        self.node_info[node].supported = False
        self.node_info[node].unsupported_reason = reason
        self._mark_node_handled(node)
    
    def _is_handled(self, node: Node) -> bool:
        return self.node_info[node].handled

    def _mark_supported(self, nodes: List[Node] | Node, *, handler: Optional[Handler] = None) -> None:
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            self._mark_node_supported(node, handler=handler)
    
    def _mark_unsupported(self, nodes: List[Node] | Node, reason: str) -> None:
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            self._mark_node_unsupported(node, reason)

    def _make_io_slots(self):
        from torch.export.graph_signature import InputKind, OutputKind, TensorArgument, SymIntArgument 
        output_kind_targets = defaultdict(set)
        constant_tensors = []
        user_inputs = []
        user_outputs = []
        mutable_buffers = []
        for ospec in self.ep.graph_signature.output_specs:
            kind = ospec.kind
            arg = ospec.arg
            name = arg.name 
            target = ospec.target
            if target is not None:
                output_kind_targets[kind].add(target)
            if kind == OutputKind.USER_OUTPUT:
                user_outputs.append(name)
                
        for ispec in self.ep.graph_signature.input_specs:
            kind = ispec.kind
            arg = ispec.arg
            name = arg.name
            target = ispec.target
            if isinstance(arg, TensorArgument):
                if kind == InputKind.PARAMETER:
                    assert target not in output_kind_targets[OutputKind.PARAMETER_MUTATION]
                    constant_tensors.append(name)
                elif kind == InputKind.BUFFER:
                    if target in output_kind_targets[OutputKind.BUFFER_MUTATION]:
                        mutable_buffers.append(name)
                    else:
                        constant_tensors.append(name)
                elif kind == InputKind.USER_INPUT:
                    user_inputs.append(name)
                elif kind == InputKind.CONSTANT_TENSOR:
                    constant_tensors.append(name)
                else:
                    raise NotImplementedError(f"Support for input {arg} is not implemented")
            elif isinstance(arg, SymIntArgument):
                if kind == InputKind.USER_INPUT:
                    user_inputs.append(name)
                else:
                    raise NotImplementedError(f"Support for input {arg} is not implemented")
            else:
                raise NotImplementedError(f"Support for input {arg} is not implemented")
        
        for node in self.ep.graph.nodes:
            if node.op == "placeholder":
                if node.users == {}:
                    continue
                if node.name in constant_tensors:
                    slot = self.make_or_get_slot(node, id_space=IdSpace.Constant)
                elif node.name in user_inputs:
                    slot = self.make_or_get_slot(node, id_space=IdSpace.Input)
                elif node.name in mutable_buffers:
                    slot = self.make_or_get_slot(node, id_space=IdSpace.MutableBuffer)
                else:
                    raise NotImplementedError(f"Support for placeholder {node.name} is not implemented")
            elif node.op == "output":
                outs, _ = pytree.tree_flatten(node.args)
                for o in outs:
                    if isinstance(o, Node) and o.name in user_outputs:
                        self.make_or_get_slot(o, id_space=IdSpace.Output)

    def _mark_noop(self):
        """
        Mark noops and dead nodes with noop handlers.
        """
        dead = set() 
        noop_handler = REGISTRY._by_target["NOOP"].handler
        for n in reversed(self.ep.graph.nodes):
            entry = REGISTRY.get(n) 
            if entry and entry.handler == noop_handler:
                dead.add(n)

            # If all users are dead, handle as noop
            if n.op != "output" and all(user in dead for user in n.users):
                self.node_info[n].handler = noop_handler
                dead.add(n)
        
    def _mark_pattern(self, name):
        for n in self.ep.graph.nodes:
            handler = REGISTRY.get_pattern_cls(name).maybe_create(self.ep, n)
            if handler is None:
                continue
            handler.set_handlers(self)

    def build(self):
        if self.prog_json is not None:
            return self.prog_json

        self._make_io_slots()
        self._mark_noop()
        for pattern in REGISTRY.patterns():
            self._mark_pattern(pattern)

        for n in tqdm.tqdm(self.ep.graph.nodes):
            if self._is_handled(n):
                continue

            if n.op in ("placeholder", "output"):
                self._mark_supported(n)
                continue
                
            if self.node_info[n].handler is not None:
                handler = self.node_info[n].handler
                handler(self, n)
                self._mark_supported(n, handler=handler)
                continue

            entry = REGISTRY.get(n)
            if entry is None:
                msg = f"no handler for target={n.target}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slot(n)
                self._mark_unsupported(n, msg)
                continue
            try:
                out_slot = entry.handler(self, n)
                self._mark_supported(n, handler=entry.handler)
            except Exception as e:
                trace_str = traceback.format_exc()
                msg = f"{entry.handler} failed for {n.target}: {e}.\n{trace_str}"
                if n.meta.get("val", None) is not None:
                    self.slot_manager.make_or_get_slot(n)
                self._mark_unsupported(n, msg)

        self._verify_build()
        prog = self._build_json()
        return prog
        
    def _verify_build(self):
        noop_handler = REGISTRY._by_target["NOOP"].handler
        for n, info in self.node_info.items():
            assert info.handled
            assert info.remaining_reads == 0, f"Exepcted {n} to have no remaining reads, but it has {info.remaining_reads}"
            if n.op == "output":
                assert self.slot_manager.get_slot(n) is None
                continue
            if info.handler in (noop_handler, PatternHandler.deferred_handler) or n.users == {}:
                assert self.slot_manager.get_slot(n) is None, f"Did not expect node {n} handled by {info.handler} to have a slot"
            else:
                assert self.slot_manager.get_slot(n) is not None, f"Expected slot for node {n}"
    
    def _build_json(self):
        if self.prog_json is not None:
            return self.prog_json

        # Check support
        unsupported = {}
        for node, info in self.node_info.items():
            if not info.supported:
                raise ValueError(f"Found unsupported node: {node}\nReason: {info.unsupported_reason}")
        
        # Loop through all instructions and 
        # find slots that are used
        used_slots: set[Slot] = set()
        for instr in self._instrs:
            for k, v in instr.items():
                if k == "op":
                    continue
                flat_args, spec = pytree.tree_flatten(v)
                for a in flat_args:
                    if isinstance(a, Slot):
                        used_slots.add(a)

        # Count used tensors/values
        num_tensors: Dict[IdSpace, int] = defaultdict(int)
        num_values: Dict[IdSpace, int] = defaultdict(int)        
        seen: set[Slot] = set()
        for n, slot in self.slot_manager.name_to_slot.items():
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                if s not in used_slots:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                if s.id_type == IdType.Tensor:
                    num_tensors[s.id_space] += 1
                else:
                    num_values[s.id_space] += 1

        id_space_order = {
            IdSpace.Constant: 0,
            IdSpace.Input: 1,
            IdSpace.Output: 2,
            IdSpace.MutableBuffer: 3,
            IdSpace.Temp: 4,
        }

        slot_to_tid = sorted(
            [s for s in used_slots if s.id_type == IdType.Tensor], 
            key=lambda s: (id_space_order[s.id_space], s.idx)
        )
        slot_to_tid = {s:idx for idx, s in enumerate(slot_to_tid)}


        slot_to_vid = sorted(
            [s for s in used_slots if s.id_type != IdType.Tensor], 
            key=lambda s: (id_space_order[s.id_space], s.idx)
        )
        slot_to_vid = {s:idx for idx, s in enumerate(slot_to_vid)}

        assert len(slot_to_tid) == sum(num_tensors.values())
        assert len(slot_to_vid) == sum(num_values.values()) 

        def serialize_value(v):
            if isinstance(v, Slot):
                if v.id_type == IdType.Tensor:
                    return {"tid": slot_to_tid[v]}
                else:
                    return {"vid": slot_to_vid[v]}
            else:
               return v

        # Create program json
        instructions = []
        for instr in self._instrs:
            payload = {}
            for k, v in instr.items():
                if k == "op":
                    continue
                flat_args, spec = pytree.tree_flatten(v)
                flat_args = [serialize_value(a) for a in flat_args]
                payload[k] = pytree.tree_unflatten(flat_args, spec)
            instructions.append({"op": instr["op"], **payload})
        
        num_constant_tensors = num_tensors[IdSpace.Constant]
        num_non_constant_tensors = sum(num_tensors.values()) - num_constant_tensors
        assert num_values[IdSpace.Constant] == 0, f"Expected no constant values, but got {num_values[IdSpace.Constant]}"
        num_non_constant_values = sum(num_values.values()) - num_values[IdSpace.Constant]
        program = {
            "num_constant_tensors": num_constant_tensors,
            "num_non_constant_tensors": num_non_constant_tensors,
            "num_non_constant_values": num_non_constant_values,
            "code": instructions,
        }
        
        def to_slot_variant(slot: Slot):
            if slot.id_type == IdType.Tensor:
                idx = slot_to_tid[slot]
                variant = "tid"
            elif slot.id_type == IdType.SymInt:
                idx = slot_to_vid[slot]
                variant = "vid[int]"
            elif slot.id_type == IdType.SymBool:
                idx = slot_to_vid[slot]
                variant = "vid[bool]"
            else:
                raise NotImplementedError(f"Unsupported slot type {slot.id_type}")

            return {"idx": idx, "variant": variant}
        
        def to_tensor_meta(t):
            return {
                "shape": [int(i) for i in t.shape],
                "dtype": str(S._TORCH_DTYPE_TO_DTYPEID[t.dtype]),
                "strides": [int(i) for i in t.stride()],
            }
        

        tensor_meta = {}        
        for n in self.node_info:
            slot = self.slot_manager.get_slot(n)
            if not isinstance(slot, tuple):
                slot = (slot,)
            for s in slot:
                if s not in used_slots:
                    continue
                if s.id_type != IdType.Tensor:
                    continue
                if s.id_space == IdSpace.Temp:
                    continue
                idx = slot_to_tid[s]
                fake_tensor = n.meta.get("val", None)
                assert fake_tensor is not None, f"Expected node {n} to have a fake tensor"
                tensor_meta[idx] = to_tensor_meta(fake_tensor)
        
        for name, t in self.extra_constants.items():
            slot = self.slot_manager.get_slot(name)
            assert slot is not None
            assert isinstance(slot, Slot)
            if slot not in used_slots:
                continue
            idx = slot_to_tid[slot]
            tensor_meta[idx] = to_tensor_meta(t)

        num_non_temp_tensors = sum(num_tensors.values()) - num_tensors[IdSpace.Temp]
        tensor_meta = [tensor_meta[i] for i in range(num_non_temp_tensors)]
        program["tensor_meta"] = tensor_meta
        
        input_map = []
        output_map = []
        mutable_buffer_map = []
        name_to_slot = {}
        for ispec in self.ep.graph_signature.input_specs:
            slot = self.slot_manager.get_slot(ispec.arg.name)
            if slot is None:
                continue
            
            # We do not expect a tuple of slots for input nodes
            assert isinstance(slot, Slot)

            name = ispec.target if ispec.target is not None else ispec.arg.name
            if slot.id_space == IdSpace.Input:
                input_map.append(to_slot_variant(slot))
                assert name not in name_to_slot
                name_to_slot[name] = slot
            elif slot.id_space == IdSpace.MutableBuffer:
                mutable_buffer_map.append(to_slot_variant(slot))
                assert name not in name_to_slot
                name_to_slot[name] = slot
            else:
                # For non-IO (parameter/buffer), we only store if its used
                if slot in used_slots:
                    assert name not in name_to_slot
                    name_to_slot[name] = slot
        
        for ospec in self.ep.graph_signature.output_specs:
            name = ospec.arg.name
            slot = self.slot_manager.get_slot(name)
            if slot is None:
                continue

            # We do not expect a tuple of slots for output nodes
            assert isinstance(slot, Slot)
            if slot.id_space == IdSpace.Output:
                output_map.append(to_slot_variant(slot))
                name = ospec.target if ospec.target is not None else ospec.arg.name
                assert name not in name_to_slot
                name_to_slot[name] = slot
            else:
                continue
        
        for name in self.extra_constants:
            slot = self.slot_manager.get_slot(name)
            assert slot is not None
            assert isinstance(slot, Slot)
            if slot in used_slots:
                assert name not in name_to_slot
                name_to_slot[name] = slot 
        
        program["input_map"] = input_map
        program["output_map"] = output_map
        program["mutable_buffer_map"] = mutable_buffer_map
        program["name_to_slot"] = {n:to_slot_variant(name_to_slot[n]) for n in name_to_slot}

        self.prog_json = json.dumps(program, indent=2)
        return self.prog_json
    
    def save_constant_data(self, filename="consts.safetensors"):
        assert self.prog_json is not None, "You need to build the porgram before saving the data"
        prog = json.loads(self.prog_json)
        
        from safetensors.torch import save_file
        tensor_dict = {}
        for name, v in self.ep.state_dict.items():
            if isinstance(v, torch.Tensor) and name in prog["name_to_slot"]:
                assert name not in tensor_dict
                tensor_dict[name] = v.detach().clone().cpu()
        
        for name, v in self.extra_constants.items():
            if isinstance(v, torch.Tensor) and name in prog["name_to_slot"]:
                assert name not in tensor_dict
                tensor_dict[name] = v.detach().clone().cpu()

        save_file(tensor_dict, filename)
