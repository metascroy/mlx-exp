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

def _is_list_of_union_int_vid(ann) -> bool:
    # List[Union[int, Vid[int]]]
    if get_origin(ann) in (list, List):
        (elem,) = get_args(ann)
        return _is_union_int_vid(elem)
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
    elif _is_list_of_union_int_vid(base):
        c = "std::vector<std::variant<int, Vid<int>>>"
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
# C++ JSON parser generation for program_json_loader.hpp
# -----------------------------------------------------------------------------

def _gen_field_parser(field_name: str, resolved_ann: Any, indent: str = "        ") -> str:
    """Generate C++ code to parse a single field from JSON."""
    opt = _is_optional(resolved_ann)
    base = _unwrap_optional(resolved_ann) if opt else resolved_ann
    
    lines = []
    json_accessor = f'jinstr.at("{field_name}")'
    
    if opt:
        # Optional fields: check if exists and not null
        lines.append(f'{indent}if (jinstr.contains("{field_name}") && !{json_accessor}.is_null()) {{')
        inner_indent = indent + "  "
        
        if _is_tid(base):
            lines.append(f'{inner_indent}n.{field_name} = detail::parse_tid_obj({json_accessor});')
        elif _is_vid(base):
            lines.append(f'{inner_indent}n.{field_name} = detail::parse_vid_int_obj({json_accessor});')
        elif base is float:
            lines.append(f'{inner_indent}n.{field_name} = {json_accessor}.get<float>();')
        elif base is int:
            lines.append(f'{inner_indent}n.{field_name} = {json_accessor}.get<int>();')
        elif base is bool:
            lines.append(f'{inner_indent}n.{field_name} = detail::parse_bool({json_accessor});')
        elif base is str:
            lines.append(f'{inner_indent}n.{field_name} = {json_accessor}.get<std::string>();')
        elif _is_dtype_enum(base):
            lines.append(f'{inner_indent}n.{field_name} = detail::parse_dtype({json_accessor}.get<std::string>());')
        elif _is_list_of_int(base):
            lines.append(f'{inner_indent}n.{field_name} = {json_accessor}.get<std::vector<int>>();')
        elif _is_list_of_str(base):
            lines.append(f'{inner_indent}n.{field_name} = {json_accessor}.get<std::vector<std::string>>();')
        else:
            lines.append(f'{inner_indent}// TODO: unsupported optional type for {field_name}')
        
        lines.append(f'{indent}}} else {{')
        lines.append(f'{inner_indent}n.{field_name} = std::nullopt;')
        lines.append(f'{indent}}}')
    else:
        # Required fields
        if _is_tid(base):
            lines.append(f'{indent}n.{field_name} = detail::parse_tid_obj({json_accessor});')
        elif _is_vid(base):
            lines.append(f'{indent}n.{field_name} = detail::parse_vid_int_obj({json_accessor});')
        elif base is float:
            lines.append(f'{indent}n.{field_name} = {json_accessor}.get<float>();')
        elif base is int:
            lines.append(f'{indent}n.{field_name} = detail::parse_int_strict({json_accessor});')
        elif base is bool:
            lines.append(f'{indent}n.{field_name} = detail::parse_bool({json_accessor});')
        elif base is str:
            lines.append(f'{indent}n.{field_name} = {json_accessor}.get<std::string>();')
        elif _is_dtype_enum(base):
            # Handle potential null for dtype with default
            lines.append(f'{indent}auto {field_name}_json = {json_accessor};')
            lines.append(f'{indent}std::string {field_name}_string = "DTypeId.i32";')
            lines.append(f'{indent}if (!{field_name}_json.is_null()) {{')
            lines.append(f'{indent}  {field_name}_string = {field_name}_json.get<std::string>();')
            lines.append(f'{indent}}}')
            lines.append(f'{indent}n.{field_name} = detail::parse_dtype({field_name}_string);')
        elif _is_list_of_int(base):
            lines.append(f'{indent}n.{field_name} = {json_accessor}.get<std::vector<int>>();')
        elif _is_list_of_str(base):
            lines.append(f'{indent}n.{field_name} = {json_accessor}.get<std::vector<std::string>>();')
        elif _is_list_of_union_int_vid(base):
            lines.append(f'{indent}n.{field_name} = detail::parse_shape_list({json_accessor});')
        elif _is_union_int_vid(base):
            lines.append(f'{indent}n.{field_name} = detail::parse_int_or_vid({json_accessor});')
        else:
            lines.append(f'{indent}// TODO: unsupported type for {field_name}')
    
    return "\n".join(lines)

def gen_json_parser() -> str:
    """Generate the JSON parsing code for all ops."""
    L: List[str] = []
    
    # Generate the if-else chain for parsing ops
    items = list(S.OPS.items())
    
    for i, (opname, cls) in enumerate(items):
        if i == 0:
            L.append(f'      if (op == "{opname}") {{')
        else:
            L.append(f'      else if (op == "{opname}") {{')
        
        # Get fields for this op
        fs = fields(cls)
        
        if not fs:
            # No fields - just create empty node
            L.append(f'        P.code.push_back(make_{opname}({cls.__name__}{{}}));')
        else:
            # Has fields - generate parsing code
            L.append(f'        {cls.__name__} n;')
            
            resolved = _resolved_ann_map(cls)
            for f in fs:
                ann = resolved.get(f.name, f.type)
                parser_code = _gen_field_parser(f.name, ann)
                L.append(parser_code)
            
            L.append(f'        P.code.push_back(make_{opname}(std::move(n)));')
        
        L.append('      }')
    
    # Add the else clause for unknown ops
    L.append('      else {')
    L.append('        throw std::runtime_error("program_from_json: unknown op " + op);')
    L.append('      }')
    
    return "\n".join(L)

def gen_program_json_loader_hpp() -> str:
    """Generate the complete program_json_loader.hpp file with auto-generated parsers."""
    L: List[str] = []
    
    # Header and includes
    L.append("// AUTO-GENERATED FILE — do not edit by hand")
    L.append("// Generated by generate_from_schema.py based on ops_schema.py")
    L.append("// program_json_loader.hpp")
    L.append("#pragma once")
    L.append('#include "program.hpp"')
    L.append('#include "ops.hpp"')
    L.append("#include <nlohmann/json.hpp>")
    L.append("#include <stdexcept>")
    L.append("#include <string>")
    L.append("#include <unordered_map>")
    L.append("#include <variant>")
    L.append("#include <vector>")
    L.append("")
    L.append("#include <mlx/array.h>")
    L.append("#include <mlx/mlx.h>")
    L.append("")
    L.append("namespace executorch::mlx {")
    L.append("")
    L.append("namespace detail {")
    L.append("")
    L.append("// ------------------------------")
    L.append("// Small helpers (strict version)")
    L.append("// ------------------------------")
    L.append("inline DTypeId parse_dtype(const std::string& s) {")
    for name in S.DTypeId.__members__:
        L.append(f'  if (s == "DTypeId.{name}") return DTypeId::{name};')
    L.append('  throw std::runtime_error("parse_dtype: unknown dtype: " + s);')
    L.append("}")
    L.append("")
    L.append("// STRICT: tensors in op payloads must be {\"tid\": N}")
    L.append("inline Tid parse_tid_obj(const nlohmann::json& j) {")
    L.append("  // Handle both {\"tid\": N} and [{\"tid\": N}] formats")
    L.append("  if (j.is_array() && j.size() == 1) {")
    L.append("    const auto& elem = j[0];")
    L.append('    if (!elem.is_object() || !elem.contains("tid"))')
    L.append('      throw std::runtime_error("parse_tid_obj: expected [{\\"tid\\": N}]");')
    L.append("    Tid t{};")
    L.append('    t.idx = elem.at("tid").get<uint32_t>();')
    L.append("    return t;")
    L.append("  }")
    L.append('  if (!j.is_object() || !j.contains("tid"))')
    L.append('    throw std::runtime_error("parse_tid_obj: expected {\\"tid\\": N}");')
    L.append("  Tid t{};")
    L.append('  t.idx = j.at("tid").get<uint32_t>();')
    L.append("  return t;")
    L.append("}")
    L.append("")
    L.append("// STRICT: value slots in op payloads must be {\"vid\": N}")
    L.append("inline Vid<int> parse_vid_int_obj(const nlohmann::json& j) {")
    L.append('  if (!j.is_object() || !j.contains("vid"))')
    L.append('    throw std::runtime_error("parse_vid_int_obj: expected {\\"vid\\": N}");')
    L.append("  Vid<int> v{};")
    L.append('  v.idx = j.at("vid").get<uint32_t>();')
    L.append("  return v;")
    L.append("}")
    L.append("")
    L.append("// name_to_slot / input_map still use the older {idx, variant} shape")
    L.append("inline Program::SlotVariant parse_slot_variant(const nlohmann::json& j) {")
    L.append('  const auto idx = j.at("idx").get<uint32_t>();')
    L.append('  const auto& variant = j.at("variant").get<std::string>();')
    L.append('  if (variant == "tid") {')
    L.append("    Tid t{idx};")
    L.append("    return t;")
    L.append('  } else if (variant == "vid[int]") {')
    L.append("    Vid<int32_t> v{idx};")
    L.append("    return v;")
    L.append('  } else if (variant == "vid[float]") {')
    L.append("    Vid<float> v{idx};")
    L.append("    return v;")
    L.append('  } else if (variant == "vid[bool]") {')
    L.append("    Vid<bool> v{idx};")
    L.append("    return v;")
    L.append('  } else if (variant == "vid[string]") {')
    L.append("    Vid<std::string> v{idx};")
    L.append("    return v;")
    L.append("  }")
    L.append('  throw std::runtime_error("parse_slot_variant: unknown variant " + variant);')
    L.append("}")
    L.append("")
    L.append("// tolerate JSON bool or 0/1")
    L.append("inline bool parse_bool(const nlohmann::json& j) {")
    L.append("  if (j.is_boolean()) return j.get<bool>();")
    L.append("  if (j.is_number_integer()) return j.get<int>() != 0;")
    L.append('  throw std::runtime_error("parse_bool: expected bool or 0/1");')
    L.append("}")
    L.append("")
    L.append("// mixed scalar fields (like SLICE axis/start/end) are allowed to be")
    L.append("//   - literal int          -> int")
    L.append('//   - {"vid": N}           -> Vid<int>')
    L.append("//   - null                 -> default literal")
    L.append("inline std::variant<int, Vid<int>> parse_int_or_vid(")
    L.append("    const nlohmann::json& j) {")
    L.append("  if (j.is_number_integer()) {")
    L.append("    return j.get<int>();  // literal attribute")
    L.append("  }")
    L.append('  if (j.is_object() && j.contains("vid")) {')
    L.append("    Vid<int> v{};")
    L.append('    v.idx = j.at("vid").get<uint32_t>();')
    L.append("    return v;")
    L.append("  }")
    L.append('  throw std::runtime_error("parse_int_or_vid: expected int or {\\"vid\\": N}");')
    L.append("}")
    L.append("")
    L.append("inline std::vector<std::variant<int, Vid<int>>> parse_shape_list(")
    L.append("    const nlohmann::json& j) {")
    L.append("  if (!j.is_array()) {")
    L.append('    throw std::runtime_error("parse_shape_list: expected array");')
    L.append("  }")
    L.append("  std::vector<std::variant<int, Vid<int>>> out;")
    L.append("  out.reserve(j.size());")
    L.append("  for (const auto& elem : j) {")
    L.append("    // reuse existing helper")
    L.append("    out.push_back(parse_int_or_vid(elem));")
    L.append("  }")
    L.append("  return out;")
    L.append("}")
    L.append("")
    L.append("inline int parse_int_strict(const nlohmann::json& j) {")
    L.append("  if (!j.is_number_integer()) {")
    L.append('    throw std::runtime_error("parse_int_strict: expected integer");')
    L.append("  }")
    L.append("  return j.get<int>();")
    L.append("}")
    L.append("")
    L.append("} // namespace detail")
    L.append("")
    L.append("// ============================================================================")
    L.append("// Main deserializer (strict slot format)")
    L.append("// ============================================================================")
    L.append("")
    L.append("inline Program program_from_json(const nlohmann::json& jprog) {")
    L.append("  Program P;")
    L.append("")
    L.append("  // ---- basic counts ----")
    L.append('  P.num_constant_tensors     = jprog.at("num_constant_tensors").get<uint32_t>();')
    L.append('  P.num_non_constant_tensors = jprog.at("num_non_constant_tensors").get<uint32_t>();')
    L.append('  P.num_non_constant_values  = jprog.at("num_non_constant_values").get<uint32_t>();')
    L.append("")
    L.append("  // ---- tensor_meta ----")
    L.append('  if (jprog.contains("tensor_meta")) {')
    L.append('    const auto& jmeta = jprog.at("tensor_meta");')
    L.append("    P.tensor_meta.resize(jmeta.size());")
    L.append("    for (size_t i = 0; i < jmeta.size(); ++i) {")
    L.append("      const auto& jm = jmeta.at(i);")
    L.append("      TensorMeta tm;")
    L.append('      tm.shape = jm.at("shape").get<std::vector<int>>();')
    L.append("      tm.dim_order.resize(tm.shape.size());")
    L.append("      for (size_t d = 0; d < tm.dim_order.size(); ++d)")
    L.append("        tm.dim_order[d] = static_cast<int>(d);")
    L.append('      tm.dtype = detail::parse_dtype(jm.at("dtype").get<std::string>());')
    L.append("      P.tensor_meta[i] = tm;")
    L.append("    }")
    L.append("  }")
    L.append("")
    L.append("  // ---- name_to_slot ----")
    L.append('  if (jprog.contains("name_to_slot")) {')
    L.append('    const auto& jn2s = jprog.at("name_to_slot");')
    L.append("    for (auto it = jn2s.begin(); it != jn2s.end(); ++it) {")
    L.append("      const std::string& name = it.key();")
    L.append("      Program::SlotVariant slot = detail::parse_slot_variant(it.value());")
    L.append("      P.nameToSlot.emplace(name, slot);")
    L.append("    }")
    L.append("  }")
    L.append("")
    L.append("  // ---- input / output / mutable buffer maps ----")
    L.append('  if (jprog.contains("input_map")) {')
    L.append('    for (const auto& jin : jprog.at("input_map")) {')
    L.append("      P.add_input(detail::parse_slot_variant(jin));")
    L.append("    }")
    L.append("  }")
    L.append('  if (jprog.contains("output_map")) {')
    L.append('    for (const auto& jout : jprog.at("output_map")) {')
    L.append("      P.add_output(detail::parse_slot_variant(jout));")
    L.append("    }")
    L.append("  }")
    L.append('  if (jprog.contains("mutable_buffer_map")) {')
    L.append('    for (const auto& jmb : jprog.at("mutable_buffer_map")) {')
    L.append("      P.add_mutable_buffer(detail::parse_slot_variant(jmb));")
    L.append("    }")
    L.append("  }")
    L.append("")
    L.append("  // ---- code ----")
    L.append('  const auto& jcode = jprog.at("code");')
    L.append("  P.code.reserve(jcode.size());")
    L.append("")
    L.append("  for (const auto& jinstr : jcode) {")
    L.append('    const std::string op = jinstr.at("op").get<std::string>();')
    L.append("    // std::cout << \"DOING OP \" << op << std::endl;")
    L.append("")
    L.append("    // ========== AUTO-GENERATED OP PARSING ==========")
    
    # Insert the generated parser code
    L.append(gen_json_parser())
    
    L.append("  }")
    L.append("")
    L.append("  return P;")
    L.append("}")
    L.append("")
    L.append("")
    L.append("inline ::mlx::core::Dtype to_mlx(DTypeId d) {")
    L.append("  using namespace ::mlx::core;")
    L.append("  switch (d) {")
    for name in S.DTypeId.__members__:
        if name == "i8":
            L.append(f"    case DTypeId::{name}:    return int8;")
        elif name == "f16":
            L.append(f"    case DTypeId::{name}:    return float16;")
        elif name == "f32":
            L.append(f"    case DTypeId::{name}:    return float32;")
        elif name == "bf16":
            L.append(f"    case DTypeId::{name}:   return bfloat16;")
        elif name == "i32":
            L.append(f"    case DTypeId::{name}:    return int32;")
        elif name == "i64":
            L.append(f"    case DTypeId::{name}:    return int64;")
        elif name == "u32":
            L.append(f"    case DTypeId::{name}:    return uint32;")
        elif name == "u8":
            L.append(f"    case DTypeId::{name}:     return uint8;")
        elif name == "boolean":
            L.append(f"    case DTypeId::{name}:return bool_;")
    L.append("  }")
    L.append('  throw std::runtime_error("to_mlx: unknown dtype");')
    L.append("}")
    L.append("")
    L.append("// If you have a global/constexpr compute dtype, keep it here")
    L.append("static constexpr DTypeId kComputeDT = DTypeId::f32;")
    L.append("")
    L.append("")
    L.append("inline void bind_constants_from_safetensors(")
    L.append("    const std::string& path,")
    L.append("    Program& P,")
    L.append("    ConstantData& store)")
    L.append("{")
    L.append("  using namespace ::mlx::core;")
    L.append("")
    L.append("  const uint32_t nconst = P.num_constant_tensors;")
    L.append("  if (nconst == 0) {")
    L.append("    store.tensors.clear();")
    L.append("    P.constants = &store;")
    L.append("    return;")
    L.append("  }")
    L.append("")
    L.append("  // Load all tensors from safetensors")
    L.append("  // Adjust to your actual API")
    L.append("  auto tensors_pair = load_safetensors(path);")
    L.append("  const auto& tensors = tensors_pair.first;")
    L.append("")
    L.append("  // collect all constant (tid,name) pairs from nameToSlot")
    L.append("  std::vector<std::pair<uint32_t, std::string>> const_slots;")
    L.append("  const_slots.reserve(P.nameToSlot.size());")
    L.append("  for (const auto& [name, slot] : P.nameToSlot) {")
    L.append("    if (std::holds_alternative<Tid>(slot)) {")
    L.append("      Tid t = std::get<Tid>(slot);")
    L.append("      if (t.idx < nconst) {")
    L.append("        const_slots.emplace_back(t.idx, name);")
    L.append("      }")
    L.append("    }")
    L.append("  }")
    L.append("")
    L.append("  // we expect to have at least all constants named,")
    L.append("  // but to be robust, sort and check gaps")
    L.append("  std::sort(const_slots.begin(), const_slots.end(),")
    L.append("            [](auto& a, auto& b){ return a.first < b.first; });")
    L.append("")
    L.append("  // rebuild constant area")
    L.append("  store.tensors.clear();")
    L.append("  store.tensors.reserve(nconst);")
    L.append("")
    L.append("  auto to_mlx_dtype = [](DTypeId d){ return to_mlx(d); };")
    L.append("")
    L.append("  auto to_target = [&](const array& ain,")
    L.append("                     DTypeId target_dt,")
    L.append("                     bool tpose = false) -> array {")
    L.append("    using namespace ::mlx::core;")
    L.append("    array a = ain;")
    L.append("")
    L.append("    // if we need to transpose, do it first")
    L.append("    if (tpose && a.ndim() == 2)")
    L.append("      a = contiguous(transpose(a, {1, 0}));")
    L.append("")
    L.append("    // for pure integer / packed types, DO NOT route through float32")
    L.append("    if (target_dt == DTypeId::u32 || target_dt == DTypeId::u8 || target_dt == DTypeId::i32) {")
    L.append("      if (a.dtype() != to_mlx(target_dt))")
    L.append("        a = astype(a, to_mlx(target_dt));")
    L.append("      return contiguous(a);")
    L.append("    }")
    L.append("")
    L.append("    // existing float path")
    L.append("    if (a.dtype() != float32)")
    L.append("      a = astype(a, float32);")
    L.append("    a = contiguous(a);")
    L.append("    if (target_dt != DTypeId::f32)")
    L.append("      a = astype(a, to_mlx(target_dt));")
    L.append("    return a;")
    L.append("  };")
    L.append("")
    L.append("")
    L.append("  // now fill in order 0..nconst-1")
    L.append("  uint32_t next_expected = 0;")
    L.append("  for (const auto& [tid_idx, name] : const_slots) {")
    L.append("    if (tid_idx != next_expected) {")
    L.append("      // JSON said we have e.g. 10 constants but we only found names for some")
    L.append("      throw std::runtime_error(")
    L.append('        "bind_constants_from_safetensors: missing constant for Tid " +')
    L.append("        std::to_string(next_expected));")
    L.append("    }")
    L.append("")
    L.append("    auto it = tensors.find(name);")
    L.append("    if (it == tensors.end()) {")
    L.append('      throw std::runtime_error("bind_constants_from_safetensors: missing key in safetensors: " + name);')
    L.append("    }")
    L.append("")
    L.append("    // pick dtype from tensor_meta if available")
    L.append("    DTypeId target_dt = DTypeId::f32;")
    L.append("    if (tid_idx < P.tensor_meta.size() && P.tensor_meta[tid_idx].has_value()) {")
    L.append("      target_dt = P.tensor_meta[tid_idx]->dtype;")
    L.append("    }")
    L.append("")
    L.append("    array cooked = to_target(it->second, target_dt, /*tpose=*/false);")
    L.append("    store.add(std::move(cooked));")
    L.append("")
    L.append("    ++next_expected;")
    L.append("  }")
    L.append("")
    L.append("  // if we didn't cover all declared constants, complain")
    L.append("  if (next_expected != nconst) {")
    L.append("    throw std::runtime_error(")
    L.append('      "bind_constants_from_safetensors: program declares " +')
    L.append("      std::to_string(nconst) +")
    L.append('      " constant tensors, but only " +')
    L.append("      std::to_string(next_expected) +")
    L.append('      " were bound by name");')
    L.append("  }")
    L.append("")
    L.append("  // hook it up")
    L.append("  P.bind_constants(store);")
    L.append("}")
    L.append("")
    L.append("inline void init_execution_state_from_meta(const Program& P, ExecutionState& S) {")
    L.append("  if (S.P != &P)")
    L.append('    throw std::runtime_error("init_execution_state_from_meta: state not bound to this Program");')
    L.append("")
    L.append("  using namespace ::mlx::core;")
    L.append("")
    L.append("  const auto n_const = P.num_constant_tensors;")
    L.append("  const auto n_total = P.num_tensors();")
    L.append("")
    L.append("  // tensor_meta can be smaller than total tensors")
    L.append("  for (uint32_t tidx = n_const; tidx < n_total; ++tidx) {")
    L.append("    const uint32_t slot = tidx - n_const;  // index into S.tensors")
    L.append("    if (tidx < P.tensor_meta.size() && P.tensor_meta[tidx].has_value()) {")
    L.append("      const auto& tm = *P.tensor_meta[tidx];")
    L.append("")
    L.append("      // build MLX shape")
    L.append("      std::vector<int64_t> shape64;")
    L.append("      shape64.reserve(tm.shape.size());")
    L.append("      for (int d : tm.shape) {")
    L.append("        shape64.push_back(static_cast<int64_t>(d));")
    L.append("      }")
    L.append("")
    L.append("      auto dtype = to_mlx(tm.dtype);")
    L.append("")
    L.append("      // allocate zeros (you can switch to uninitialized if MLX exposes it)")
    L.append("      array a = zeros(::mlx::core::Shape(shape64.begin(), shape64.end()), dtype);")
    L.append("      S.tensors[slot] = std::move(a);")
    L.append("    } else {")
    L.append("      // leave as std::nullopt -> user/interpreter will fill it later")
    L.append("    }")
    L.append("  }")
    L.append("}")
    L.append("")
    L.append("")
    L.append("} // namespace executorch::mlx")
    L.append("")
    return "\n".join(L)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    _assert_field_order()
    with open("generated_ops_mixin.py", "w", encoding="utf-8") as f:
        f.write(gen_ops_mixin_py())
    with open("src/ops.hpp", "w", encoding="utf-8") as f:
        f.write(gen_ops_hpp())
    with open("src/program_json_loader.hpp", "w", encoding="utf-8") as f:
        f.write(gen_program_json_loader_hpp())
    print(f"Wrote generated_ops_mixin.py, src/ops.hpp, and src/program_json_loader.hpp ({len(S.OPS)} ops)")

if __name__ == "__main__":
    main()
