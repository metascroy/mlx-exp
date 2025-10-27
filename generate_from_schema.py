#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# generate_from_schema.py — emits generated_ops_mixin.py and ops.hpp
# Matches legacy ops.hpp structure: OP_LIST, NodeVariant, Instr, factories
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import ops_schema as S

# -----------------------------------------------------------------------------
# Annotation helpers
# -----------------------------------------------------------------------------

def _is_optional(ann) -> bool:
    return get_origin(ann) is Union and type(None) in get_args(ann)

def _unwrap_optional(ann):
    return next((a for a in get_args(ann) if a is not type(None)), Any)

def _is_list_of_int(ann) -> bool:
    return get_origin(ann) in (list, List) and get_args(ann) == (int,)

def _is_list_of_str(ann) -> bool:
    return get_origin(ann) in (list, List) and get_args(ann) == (str,)

def _is_dtype_enum(ann) -> bool:
    return ann is S.DTypeId

def _is_tid(ann) -> bool:
    return ann is S.Tid or getattr(ann, "__name__", "") == "Tid"

def _is_vid(ann) -> bool:
    # Vid[T] or Vid
    if ann is S.Vid:
        return True
    if getattr(ann, "__origin__", None) is S.Vid:
        return True
    return getattr(ann, "__name__", "") == "Vid"

def _vid_cpp_type(ann: Any) -> str:
    """Return the C++ type string for Vid[T]."""
    # Parameterized Vid[T]
    if getattr(ann, "__origin__", None) is S.Vid:
        args = get_args(ann)
        if args:
            t = args[0]
            if t is int:
                return "Vid<int>"
            if t is float:
                return "Vid<float>"
            if t is bool:
                return "Vid<bool>"
            if t is str:
                return "Vid<std::string>"
        return "Vid<void>"
    # Bare Vid in annotations (shouldn’t normally happen)
    return "Vid<void>"

def _is_union_int_vid(ann) -> bool:
    """Detect Union[int, Vid[int]] (used in SLICE/SLICE_UPDATE axis/start/length)."""
    if get_origin(ann) is Union:
        args = get_args(ann)
        if len(args) == 2 and int in args:
            other = args[0] if args[1] is int else args[1]
            return _is_vid(other)
    return False

def _resolved_ann_map(cls) -> Dict[str, Any]:
    return get_type_hints(cls, globalns=vars(S), localns=vars(S))

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

def _has_default(f) -> bool:
    return f.default is not MISSING

def _has_default_factory(f) -> bool:
    return getattr(f, "default_factory", MISSING) is not MISSING

def _py_default_repr(f):
    """Render Python default value for mixin signature."""
    if _has_default(f):
        dv = f.default
        if dv is None:
            return "None"
        if isinstance(dv, S.DTypeId):
            return f"'{dv.name}'"
        if isinstance(dv, (int, float)):
            return repr(dv)
        if isinstance(dv, bool):
            return "True" if dv else "False"
        if isinstance(dv, str):
            return repr(dv)
        return None
    if _has_default_factory(f):
        return None
    return None

# -----------------------------------------------------------------------------
# C++ type/default helpers
# -----------------------------------------------------------------------------

def _ctype(resolved_ann: Any) -> str:
    opt = _is_optional(resolved_ann)
    base = _unwrap_optional(resolved_ann) if opt else resolved_ann

    if base is int:
        c = "int"
    elif base is float:
        c = "float"
    elif base is bool:
        c = "bool"
    elif base is str:
        c = "std::string"
    elif _is_list_of_int(base):
        c = "std::vector<int>"
    elif _is_list_of_str(base):
        c = "std::vector<std::string>"
    elif _is_dtype_enum(base):
        c = "DTypeId"
    elif _is_tid(base):
        c = "Tid"
    elif _is_vid(base):
        c = _vid_cpp_type(base)
    elif _is_union_int_vid(base):
        c = "std::variant<int, Vid<int>>"
    else:
        raise RuntimeError(f"Unmapped type in C++ codegen: {base}")
    return f"std::optional<{c}>" if opt else c

def _cxx_default(f, resolved_ann: Any) -> Optional[str]:
    opt = _is_optional(resolved_ann)
    base = _unwrap_optional(resolved_ann) if opt else resolved_ann

    if opt:
        if _has_default(f) and f.default is not None:
            dv = f.default
            if base is float:
                return f"{float(dv)}f"
            if base is int:
                return str(int(dv))
            if base is bool:
                return "true" if dv else "false"
            if base is str:
                return f"\"{dv}\""
            if _is_dtype_enum(base):
                return f"DTypeId::{dv.name}"
            # Optional Tid/Vid defaults other than None are unusual; omit
            return None
        return "std::nullopt"

    if base is int:
        return str(int(f.default)) if _has_default(f) else None
    if base is float:
        return f"{float(f.default)}f" if _has_default(f) else None
    if base is bool:
        return ("true" if f.default else "false") if _has_default(f) else None
    if base is str:
        return f"\"{f.default}\"" if _has_default(f) else None
    if _is_dtype_enum(base):
        return f"DTypeId::{f.default.name}" if _has_default(f) else None
    # default-initialize ids/variants
    if _is_tid(base):
        return "Tid{}"
    if _is_vid(base):
        return _vid_cpp_type(base) + "{}"
    if _is_union_int_vid(base):
        return "std::variant<int, Vid<int>>{}"
    return None

# -----------------------------------------------------------------------------
# Python typing hint helper (for mixin)
# -----------------------------------------------------------------------------

def _py_hint_for_ann(ann) -> str:
    """Return a Python type hint string for the generated mixin."""
    is_opt = _is_optional(ann)
    base = _unwrap_optional(ann) if is_opt else ann

    if base in (int, float, bool, str):
        h = base.__name__
    elif _is_list_of_int(base):
        h = "List[int]"
    elif _is_list_of_str(base):
        h = "List[str]"
    elif _is_dtype_enum(base):
        # Accept either enum or string when calling the mixin; normalize later
        h = "DTypeId | str"
    elif _is_tid(base) or _is_vid(base):
        h = "Any"
    elif _is_union_int_vid(base):
        h = "int | Any"
    else:
        h = "Any"

    return f"Optional[{h}]" if is_opt else h

# -----------------------------------------------------------------------------
# Sanity: dataclass field order
# -----------------------------------------------------------------------------

def _assert_field_order():
    for opname, cls in S.OPS.items():
        if not is_dataclass(cls):
            continue
        saw_default = False
        for f in fields(cls):
            if _has_default(f) or _has_default_factory(f):
                saw_default = True
            elif saw_default:
                raise RuntimeError(
                    f"{opname}.{f.name}: non-default field follows a default field"
                )

# -----------------------------------------------------------------------------
# Python mixin generation
# -----------------------------------------------------------------------------

def gen_ops_mixin_py() -> str:
    L: List[str] = []
    L.append("# AUTO-GENERATED FILE — do not edit by hand")
    L.append("# Generated by generate_from_schema.py based on ops_schema.py")
    L.append("from __future__ import annotations")
    L.append("from typing import Optional, List, Any, Dict")
    L.append("from ops_schema import DTypeId")
    L.append("")
    L.append("class OpsMixin:")
    L.append('    """')
    L.append("    Generated mixin with one method per op.")
    L.append("    Subclasses must implement:")
    L.append("      - _coerce_payload(self, op: str, payload: Dict[str, Any]) -> Dict[str, Any]")
    L.append("      - _emit(self, op: str, **payload) -> None")
    L.append('    """')
    L.append("")
    for opname, cls in S.OPS.items():
        fs = fields(cls)
        resolved = _resolved_ann_map(cls)
        if not fs:
            L.append(f"    def {opname}(self) -> None:")
            L.append(f"        self._emit(\"{opname}\")")
            L.append("")
            continue
        args = []
        for f in fs:
            ann = resolved.get(f.name, f.type)
            hint = _py_hint_for_ann(ann)
            default = _py_default_repr(f)
            if default is not None:
                args.append(f"{f.name}: {hint} = {default}")
            else:
                args.append(f"{f.name}: {hint}")
        sig = ", ".join(["*", *args])
        L.append(f"    def {opname}(self, {sig}) -> None:")
        L.append(f"        payload = self._coerce_payload(\"{opname}\", locals())")
        L.append(f"        self._emit(\"{opname}\", **payload)")
        L.append("")
    return "\n".join(L)

# -----------------------------------------------------------------------------
# C++ ops.hpp generation (legacy structure with OP_LIST/Instr/etc.)
# -----------------------------------------------------------------------------

def gen_ops_hpp() -> str:
    L: List[str] = []
    L.append("// AUTO-GENERATED FILE — do not edit by hand")
    L.append("// Generated by generate_from_schema.py based on ops_schema.py")
    L.append("#pragma once")
    L.append("#include <cstdint>")
    L.append("#include <cstddef>")
    L.append("#include <optional>")
    L.append("#include <variant>")
    L.append("#include <vector>")
    L.append("#include <string>")
    L.append("#include <utility>")
    L.append("")
    L.append("struct Tid { uint32_t idx{}; };")
    L.append("template <typename T>")
    L.append("struct Vid { uint32_t idx{}; };")
    L.append("")
    L.append("enum class DTypeId : int {")
    for name in S.DTypeId.__members__:
        L.append(f"  {name},")
    L.append("};")
    L.append("")
    L.append("// -----------------------------------------------------------------------------")
    L.append("// Per-op payloads (schemas)")
    L.append("// -----------------------------------------------------------------------------")

    # Collect items in stable order
    items = list(S.OPS.items())

    for opname, cls in items:
        L.append(f"struct {cls.__name__} {{")
        resolved = _resolved_ann_map(cls)
        for f in fields(cls):
            ann = resolved.get(f.name, f.type)
            ctype = _ctype(ann)
            default = _cxx_default(f, ann)
            if default is not None:
                L.append(f"  {ctype} {f.name} {{ {default} }};")
            else:
                L.append(f"  {ctype} {f.name} {{}};")
        L.append("};")
        L.append("")

    # X-macro list: OP_LIST
    L.append("// -----------------------------------------------------------------------------")
    L.append("// X-macro master list (NAME, PAYLOAD_TYPE)")
    L.append("// -----------------------------------------------------------------------------")
    L.append("#ifndef OP_LIST")
    L.append("#define OP_LIST(X) \\")
    for i, (opname, cls) in enumerate(items):
        sep = " \\"
        if i == len(items) - 1:
            sep = ""
        L.append(f"  X({opname}, {cls.__name__}){sep}")
    L.append("#endif")
    L.append("")
    # OpCode
    L.append("enum class OpCode : uint8_t {")
    L.append("#define DEFINE_ENUM(NAME, PAYLOAD) NAME,")
    L.append("  OP_LIST(DEFINE_ENUM)")
    L.append("#undef DEFINE_ENUM")
    L.append("  SENTINEL")
    L.append("};")
    L.append("")
    # Traits
    L.append("template <OpCode> struct OpPayload;")
    L.append("#define DEFINE_TRAIT(NAME, PAYLOAD) \\")
    L.append("  template <> struct OpPayload<OpCode::NAME> { using type = PAYLOAD; };")
    L.append("OP_LIST(DEFINE_TRAIT)")
    L.append("#undef DEFINE_TRAIT")
    L.append("")
    L.append("template <OpCode OC>")
    L.append("using OpPayloadT = typename OpPayload<OC>::type;")
    L.append("")
    # NodeVariant
    L.append("// NodeVariant (allows duplicate payload types via index-based emplace)")
    L.append("using NodeVariant = std::variant<")
    for opname, cls in items:
        L.append(f"  {cls.__name__},")
    L.append("  std::monostate")
    L.append(">;")
    L.append("")
    # Variant index enum
    L.append("enum : size_t {")
    for opname, cls in items:
        L.append(f"  VAR_IDX_{opname},")
    L.append("  VAR_IDX_SENTINEL")
    L.append("};")
    L.append("")
    # OpVariantIndex traits
    L.append("template <OpCode> struct OpVariantIndex;")
    for opname, cls in items:
        L.append(f"template <> struct OpVariantIndex<OpCode::{opname}> "
                 f"{{ static constexpr size_t value = VAR_IDX_{opname}; }};")
    L.append("")
    L.append("static_assert(std::variant_size<NodeVariant>::value >= VAR_IDX_SENTINEL,")
    L.append("              \"NodeVariant must have at least as many alts as ops\");")
    L.append("")
    # kOpName
    L.append("static constexpr const char* kOpName[static_cast<size_t>(OpCode::SENTINEL)] = {")
    L.append("#define NAME_ROW(NAME, PAYLOAD) #NAME,")
    L.append("  OP_LIST(NAME_ROW)")
    L.append("#undef NAME_ROW")
    L.append("};")
    L.append("static_assert(sizeof(kOpName) / sizeof(kOpName[0]) ==")
    L.append("              static_cast<size_t>(OpCode::SENTINEL),")
    L.append("              \"kOpName size must match OpCode::SENTINEL\");")
    L.append("")
    # Instr
    L.append("// Instruction type w/ index-based emplace for duplicate payloads")
    L.append("struct Instr {")
    L.append("  OpCode      op{OpCode::NOOP};")
    L.append("  NodeVariant node{")
    noop_cls_name = S.OPS["NOOP"].__name__ if "NOOP" in S.OPS else items[0][1].__name__
    L.append(f"    {noop_cls_name}{{}}")
    L.append("  };")
    L.append("")
    L.append("  Instr() = default;")
    L.append("")
    L.append("  template <OpCode OC>")
    L.append("  static Instr make(OpPayloadT<OC> payload) {")
    L.append("    Instr i;")
    L.append("    i.op = OC;")
    L.append("    i.node.template emplace<OpVariantIndex<OC>::value>(std::move(payload));")
    L.append("    return i;")
    L.append("  }")
    L.append("")
    L.append("  template <class T>       T& get()       { return std::get<T>(node); }")
    L.append("  template <class T> const T& get() const { return std::get<T>(node); }")
    L.append("")
    L.append("  template <class F> decltype(auto) visit(F&& f)       { return std::visit(std::forward<F>(f), node); }")
    L.append("  template <class F> decltype(auto) visit(F&& f) const { return std::visit(std::forward<F>(f), node); }")
    L.append("};")
    L.append("")
    # Sanity count
    L.append("static_assert(static_cast<size_t>(OpCode::SENTINEL) == ([]{")
    L.append("  size_t n = 0;")
    L.append("#define COUNT_ONE(NAME, PAYLOAD) ++n;")
    L.append("  OP_LIST(COUNT_ONE)")
    L.append("#undef COUNT_ONE")
    L.append("  return n;")
    L.append("})(), \"OpCode::COUNT mismatch with OP_LIST\");")
    L.append("")
    # make_* factories
    L.append("// Auto-generated factories: make_<OP>(payload)")
    L.append("#define DEFINE_MAKE_FN(NAME, PAYLOAD) \\")
    L.append("  inline Instr make_##NAME(PAYLOAD n) { return Instr::make<OpCode::NAME>(std::move(n)); }")
    L.append("OP_LIST(DEFINE_MAKE_FN)")
    L.append("#undef DEFINE_MAKE_FN")
    L.append("")
    return "\n".join(L)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    _assert_field_order()
    with open("generated_ops_mixin.py", "w", encoding="utf-8") as f:
        f.write(gen_ops_mixin_py())
    with open("ops.hpp", "w", encoding="utf-8") as f:
        f.write(gen_ops_hpp())
    print(f"Wrote generated_ops_mixin.py and ops.hpp ({len(S.OPS)} ops)")

if __name__ == "__main__":
    main()
