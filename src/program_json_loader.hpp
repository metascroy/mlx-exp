// program_json_loader.hpp
#pragma once
#include "program.hpp"
#include "ops.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <mlx/array.h>
#include <mlx/mlx.h>

namespace executorch::mlx {

namespace detail {

// ------------------------------
// Small helpers (strict version)
// ------------------------------
inline DTypeId parse_dtype(const std::string& s) {
  if (s == "DTypeId.f16") return DTypeId::f16;
  if (s == "DTypeId.f32") return DTypeId::f32;
  if (s == "DTypeId.bf16") return DTypeId::bf16;
  if (s == "DTypeId.i32") return DTypeId::i32;
  if (s == "DTypeId.i64") return DTypeId::i64;
  if (s == "DTypeId.u32") return DTypeId::u32;
  if (s == "DTypeId.u8") return DTypeId::u8;
  if (s == "DTypeId.boolean") return DTypeId::boolean;
  throw std::runtime_error("parse_dtype: unknown dtype: " + s);
}

// STRICT: tensors in op payloads must be {"tid": N}
inline Tid parse_tid_obj(const nlohmann::json& j) {
  if (!j.is_object() || !j.contains("tid"))
    throw std::runtime_error("parse_tid_obj: expected {\"tid\": N}");
  Tid t{};
  t.idx = j.at("tid").get<uint32_t>();
  return t;
}

// STRICT: value slots in op payloads must be {"vid": N}
inline Vid<int> parse_vid_int_obj(const nlohmann::json& j) {
  if (!j.is_object() || !j.contains("vid"))
    throw std::runtime_error("parse_vid_int_obj: expected {\"vid\": N}");
  Vid<int> v{};
  v.idx = j.at("vid").get<uint32_t>();
  return v;
}

// name_to_slot / input_map still use the older {idx, variant} shape
inline Program::SlotVariant parse_slot_variant(const nlohmann::json& j) {
  const auto idx = j.at("idx").get<uint32_t>();
  const auto& variant = j.at("variant").get<std::string>();
  if (variant == "tid") {
    Tid t{idx};
    return t;
  } else if (variant == "vid[int]") {
    Vid<int32_t> v{idx};
    return v;
  } else if (variant == "vid[float]") {
    Vid<float> v{idx};
    return v;
  } else if (variant == "vid[bool]") {
    Vid<bool> v{idx};
    return v;
  } else if (variant == "vid[string]") {
    Vid<std::string> v{idx};
    return v;
  }
  throw std::runtime_error("parse_slot_variant: unknown variant " + variant);
}

// tolerate JSON bool or 0/1
inline bool parse_bool(const nlohmann::json& j) {
  if (j.is_boolean()) return j.get<bool>();
  if (j.is_number_integer()) return j.get<int>() != 0;
  throw std::runtime_error("parse_bool: expected bool or 0/1");
}

// mixed scalar fields (like SLICE axis/start/end) are allowed to be
//   - literal int          -> int
//   - {"vid": N}           -> Vid<int>
//   - null                 -> default literal
inline std::variant<int, Vid<int>> parse_int_or_vid(
    const nlohmann::json& j) {
  if (j.is_number_integer()) {
    return j.get<int>();  // literal attribute
  }
  if (j.is_object() && j.contains("vid")) {
    Vid<int> v{};
    v.idx = j.at("vid").get<uint32_t>();
    return v;
  }
  throw std::runtime_error("parse_int_or_vid: expected int or {\"vid\": N}");
}

inline std::vector<std::variant<int, Vid<int>>> parse_shape_list(
    const nlohmann::json& j) {
  if (!j.is_array()) {
    throw std::runtime_error("parse_shape_list: expected array");
  }
  std::vector<std::variant<int, Vid<int>>> out;
  out.reserve(j.size());
  for (const auto& elem : j) {
    // reuse existing helper
    out.push_back(parse_int_or_vid(elem));
  }
  return out;
}

inline int parse_int_strict(const nlohmann::json& j) {
  if (!j.is_number_integer()) {
    throw std::runtime_error("parse_int_strict: expected integer");
  }
  return j.get<int>();
}

} // namespace detail

// ============================================================================
// Main deserializer (strict slot format)
// ============================================================================

inline Program program_from_json(const nlohmann::json& jprog) {
  Program P;

  // ---- basic counts ----
  P.num_constant_tensors     = jprog.at("num_constant_tensors").get<uint32_t>();
  P.num_non_constant_tensors = jprog.at("num_non_constant_tensors").get<uint32_t>();
  P.num_non_constant_values  = jprog.at("num_non_constant_values").get<uint32_t>();

  // ---- tensor_meta ----
  if (jprog.contains("tensor_meta")) {
    const auto& jmeta = jprog.at("tensor_meta");
    P.tensor_meta.resize(jmeta.size());
    for (size_t i = 0; i < jmeta.size(); ++i) {
      const auto& jm = jmeta.at(i);
      TensorMeta tm;
      tm.shape = jm.at("shape").get<std::vector<int>>();
      tm.dim_order.resize(tm.shape.size());
      for (size_t d = 0; d < tm.dim_order.size(); ++d)
        tm.dim_order[d] = static_cast<int>(d);
      tm.dtype = detail::parse_dtype(jm.at("dtype").get<std::string>());
      P.tensor_meta[i] = tm;
    }
  }

  // ---- name_to_slot ----
  if (jprog.contains("name_to_slot")) {
    const auto& jn2s = jprog.at("name_to_slot");
    for (auto it = jn2s.begin(); it != jn2s.end(); ++it) {
      const std::string& name = it.key();
      Program::SlotVariant slot = detail::parse_slot_variant(it.value());
      P.nameToSlot.emplace(name, slot);
    }
  }

  // ---- input / output / mutable buffer maps ----
  if (jprog.contains("input_map")) {
    for (const auto& jin : jprog.at("input_map")) {
      P.add_input(detail::parse_slot_variant(jin));
    }
  }
  if (jprog.contains("output_map")) {
    for (const auto& jout : jprog.at("output_map")) {
      P.add_output(detail::parse_slot_variant(jout));
    }
  }
  if (jprog.contains("mutable_buffer_map")) {
    for (const auto& jmb : jprog.at("mutable_buffer_map")) {
      P.add_mutable_buffer(detail::parse_slot_variant(jmb));
    }
  }

  // ---- code ----
  const auto& jcode = jprog.at("code");
  P.code.reserve(jcode.size());

  for (const auto& jinstr : jcode) {
    const std::string op = jinstr.at("op").get<std::string>();
    // std::cout << "DOING OP " << op << std::endl;

    // ========== NOOP ==========
    if (op == "NOOP") {
      P.code.push_back(make_NOOP(NoopNode{}));
    }
    // ========== LINEAR ==========
    else if (op == "LINEAR") {
      LinearNode n;
      n.x      = detail::parse_tid_obj(jinstr.at("x"));
      n.weight = detail::parse_tid_obj(jinstr.at("weight"));
      n.out    = detail::parse_tid_obj(jinstr.at("out"));
      if (jinstr.contains("bias") && !jinstr.at("bias").is_null()) {
        n.bias = detail::parse_tid_obj(jinstr.at("bias"));
      } else {
        n.bias = std::nullopt;
      }
      P.code.push_back(make_LINEAR(std::move(n)));
    }
    // ========== RMS_NORM ==========
    else if (op == "RMS_NORM") {
      RMSNormNode n;
      n.x      = detail::parse_tid_obj(jinstr.at("x"));
      n.weight = detail::parse_tid_obj(jinstr.at("weight"));
      n.out    = detail::parse_tid_obj(jinstr.at("out"));
      n.eps    = jinstr.at("eps").get<float>();
      P.code.push_back(make_RMS_NORM(std::move(n)));
    }
    // ========== ROPE_APPLY ==========
    else if (op == "ROPE_APPLY") {
      RopeNode n;
      n.q_in  = detail::parse_tid_obj(jinstr.at("q_in"));
      n.k_in  = detail::parse_tid_obj(jinstr.at("k_in"));
      n.q_out = detail::parse_tid_obj(jinstr.at("q_out"));
      n.k_out = detail::parse_tid_obj(jinstr.at("k_out"));
      n.head_dim = jinstr.at("head_dim").get<int>();

      // STRICT: pos must be {"vid": N}
      n.pos = detail::parse_vid_int_obj(jinstr.at("pos"));

      if (jinstr.contains("freqs") && !jinstr.at("freqs").is_null()) {
        n.freqs = detail::parse_tid_obj(jinstr.at("freqs"));
      } else {
        n.freqs = std::nullopt;
      }
      n.traditional = detail::parse_bool(jinstr.at("traditional"));

      if (jinstr.contains("base") && !jinstr.at("base").is_null()) {
        n.base = jinstr.at("base").get<float>();
      } else {
        n.base = std::nullopt;
      }
      n.scale = jinstr.at("scale").get<float>();
      P.code.push_back(make_ROPE_APPLY(std::move(n)));
    }
    // ========== SDPA ==========
    else if (op == "SDPA") {
      SdpaNode n;
      n.q = detail::parse_tid_obj(jinstr.at("q"));
      n.k = detail::parse_tid_obj(jinstr.at("k"));
      n.v = detail::parse_tid_obj(jinstr.at("v"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.scale = jinstr.at("scale").get<float>();
      if (jinstr.contains("mask") && !jinstr.at("mask").is_null()) {
        n.mask = detail::parse_tid_obj(jinstr.at("mask"));
      } else {
        n.mask = std::nullopt;
      }
      n.causal = detail::parse_bool(jinstr.at("causal"));
      P.code.push_back(make_SDPA(std::move(n)));
    }
    // ========== ADD ==========
    else if (op == "ADD") {
      AddNode n;
      n.a   = detail::parse_tid_obj(jinstr.at("a"));
      n.b   = detail::parse_tid_obj(jinstr.at("b"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_ADD(std::move(n)));
    }
    // ========== ADD_SCALAR ==========
    else if (op == "ADD_SCALAR") {
      AddScalarNode n;
      // literals or {"vid": N}
      n.a = detail::parse_int_or_vid(jinstr.at("a"));
      n.b = detail::parse_int_or_vid(jinstr.at("b"));
      // out must be {"vid": N}
      n.out = detail::parse_vid_int_obj(jinstr.at("out"));
      P.code.push_back(make_ADD_SCALAR(std::move(n)));
    }
    // ========== MUL ==========
    else if (op == "MUL") {
      MulNode n;
      n.a   = detail::parse_tid_obj(jinstr.at("a"));
      n.b   = detail::parse_tid_obj(jinstr.at("b"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_MUL(std::move(n)));
    }
    // ========== SILU ==========
    else if (op == "SILU") {
      SiluNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_SILU(std::move(n)));
    }
    // ========== RESHAPE ==========
    else if (op == "RESHAPE") {
      ReshapeNode n;
      n.x    = detail::parse_tid_obj(jinstr.at("x"));
      n.out  = detail::parse_tid_obj(jinstr.at("out"));
      n.shape = detail::parse_shape_list(jinstr.at("shape"));
      P.code.push_back(make_RESHAPE(std::move(n)));
    }
    // ========== TRANSPOSE ==========
    else if (op == "TRANSPOSE") {
      TransposeNode n;
      n.x    = detail::parse_tid_obj(jinstr.at("x"));
      n.out  = detail::parse_tid_obj(jinstr.at("out"));
      n.perm = jinstr.at("perm").get<std::vector<int>>();
      P.code.push_back(make_TRANSPOSE(std::move(n)));
    }
    else if (op == "SYM_SIZE") {
      SymSizeNode n;
      n.a   = detail::parse_tid_obj(jinstr.at("a"));
      n.dim = detail::parse_int_strict(jinstr.at("dim"));
      n.out = detail::parse_vid_int_obj(jinstr.at("out"));
      P.code.push_back(make_SYM_SIZE(std::move(n)));
    }
    else if (op == "CONV_1D") {
      Conv1DNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.w = detail::parse_tid_obj(jinstr.at("w"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.stride = detail::parse_int_strict(jinstr.at("stride"));
      n.padding = detail::parse_int_strict(jinstr.at("padding"));
      n.dilation = detail::parse_int_strict(jinstr.at("dilation"));
      n.groups = detail::parse_int_strict(jinstr.at("groups"));
      P.code.push_back(make_CONV_1D(std::move(n)));
    }
    else if (op == "GELU") {
      GeluNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_GELU(std::move(n)));
    }
    else if (op == "ARANGE") {
      ARangeNode n;
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.start = detail::parse_int_strict(jinstr.at("start"));
      n.stop = detail::parse_int_strict(jinstr.at("stop"));
      n.step = detail::parse_int_strict(jinstr.at("step"));

      auto dtype = jinstr.at("dtype");
      std::string dtype_string = "DTypeId.i32";
      if (!dtype.is_null()) { // TODO: handle null in interpreter?
        dtype_string = dtype.get<std::string>();
      }
      n.dtype = detail::parse_dtype(dtype_string);
      P.code.push_back(make_ARANGE(std::move(n)));
    }
    else if (op == "LAYER_NORM") {
      LayerNormNode n;
      n.x = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.weight = detail::parse_tid_obj(jinstr.at("weight")); // TOOD: make sure null is handled
      n.bias = detail::parse_tid_obj(jinstr.at("bias")); // TODO: make sure null is handled
      n.eps = jinstr.at("eps").get<float>();
      P.code.push_back(make_LAYER_NORM(std::move(n)));
    }
    // ========== CONTIGUOUS ==========
    else if (op == "CONTIGUOUS") {
      ContigNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_CONTIGUOUS(std::move(n)));
    }
    // ========== GATHER ==========
    else if (op == "GATHER") {
      GatherNode n;
      n.table = detail::parse_tid_obj(jinstr.at("table"));
      n.ids   = detail::parse_tid_obj(jinstr.at("ids"));
      n.out   = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_GATHER(std::move(n)));
    }
     // ========== ID_COPY ==========
    else if (op == "ID_COPY") {
      IdCopyNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      P.code.push_back(make_ID_COPY(std::move(n)));
    }
    // ========== SLICE ==========
    else if (op == "SLICE") {
      SliceNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      // axis/start are attributes in your example, end is {"vid": 1}
      n.axis  = detail::parse_int_or_vid(jinstr.at("axis"));
      n.start = detail::parse_int_or_vid(jinstr.at("start"));
      n.end   = detail::parse_int_or_vid(jinstr.at("end"));
      P.code.push_back(make_SLICE(std::move(n)));
    }
    // ========== CAST ==========
    else if (op == "CAST") {
      CastNode n;
      n.x    = detail::parse_tid_obj(jinstr.at("x"));
      n.out  = detail::parse_tid_obj(jinstr.at("out"));
      n.dtype = detail::parse_dtype(jinstr.at("dtype").get<std::string>());
      P.code.push_back(make_CAST(std::move(n)));
    }
    // ========== QUANTIZED_LINEAR ==========
    else if (op == "QUANTIZED_LINEAR") {
      QuantizedLinearNode n;
      n.x      = detail::parse_tid_obj(jinstr.at("x"));
      n.w      = detail::parse_tid_obj(jinstr.at("w"));
      n.scales = detail::parse_tid_obj(jinstr.at("scales"));
      n.out    = detail::parse_tid_obj(jinstr.at("out"));
      if (jinstr.contains("biases") && !jinstr.at("biases").is_null()) {
        n.biases = detail::parse_tid_obj(jinstr.at("biases"));
      } else {
        n.biases = std::nullopt;
      }
      if (jinstr.contains("bias") && !jinstr.at("bias").is_null()) {
        n.bias = detail::parse_tid_obj(jinstr.at("bias"));
      } else {
        n.bias = std::nullopt;
      }
      n.group_size = jinstr.at("group_size").get<int>();
      n.bits       = jinstr.at("bits").get<int>();
      n.mode       = jinstr.at("mode").get<std::string>();
      n.out_dtype  = detail::parse_dtype(jinstr.at("out_dtype").get<std::string>());
      P.code.push_back(make_QUANTIZED_LINEAR(std::move(n)));
    }
    // ========== CONCAT ==========
    else if (op == "CONCAT") {
      ConcatNode n;
      n.a   = detail::parse_tid_obj(jinstr.at("a"));
      n.b   = detail::parse_tid_obj(jinstr.at("b"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.axis = jinstr.at("axis").get<int>();
      P.code.push_back(make_CONCAT(std::move(n)));
    }
    // ========== FULL ==========
    else if (op == "FULL") {
      FullNode n;
      n.out   = detail::parse_tid_obj(jinstr.at("out"));
      n.shape = jinstr.at("shape").get<std::vector<int>>();
      n.v     = jinstr.at("v").get<float>();
      n.dtype = detail::parse_dtype(jinstr.at("dtype").get<std::string>());
      P.code.push_back(make_FULL(std::move(n)));
    }
    // ========== ZEROS ==========
    else if (op == "ZEROS") {
      ZerosNode n;
      n.out   = detail::parse_tid_obj(jinstr.at("out"));
      n.shape = jinstr.at("shape").get<std::vector<int>>();
      n.dtype = detail::parse_dtype(jinstr.at("dtype").get<std::string>());
      P.code.push_back(make_ZEROS(std::move(n)));
    }
    // ========== ONES ==========
    else if (op == "ONES") {
      OnesNode n;
      n.out   = detail::parse_tid_obj(jinstr.at("out"));
      n.shape = jinstr.at("shape").get<std::vector<int>>();
      n.dtype = detail::parse_dtype(jinstr.at("dtype").get<std::string>());
      P.code.push_back(make_ONES(std::move(n)));
    }
    // ========== ARGMAX ==========
    else if (op == "ARGMAX") {
      ArgmaxNode n;
      n.x   = detail::parse_tid_obj(jinstr.at("x"));
      n.out = detail::parse_tid_obj(jinstr.at("out"));
      n.axis = jinstr.at("axis").get<int>();
      P.code.push_back(make_ARGMAX(std::move(n)));
    }
    // ========== SLICE_UPDATE ==========
    else if (op == "SLICE_UPDATE") {
      SliceUpdateNode n;
      n.dst    = detail::parse_tid_obj(jinstr.at("dst"));
      n.update = detail::parse_tid_obj(jinstr.at("update"));
      n.axis   = detail::parse_int_or_vid(jinstr.at("axis"));
      n.start  = detail::parse_int_or_vid(jinstr.at("start"));
      n.stop   = detail::parse_int_or_vid(jinstr.at("stop"));
      P.code.push_back(make_SLICE_UPDATE(std::move(n)));
    }
    // ========== QUANTIZED_GATHER ==========
    else if (op == "QUANTIZED_GATHER") {
      QuantizedGatherNode n;
      n.table_q = detail::parse_tid_obj(jinstr.at("table_q"));
      n.scales  = detail::parse_tid_obj(jinstr.at("scales"));
      n.ids     = detail::parse_tid_obj(jinstr.at("ids"));
      n.out     = detail::parse_tid_obj(jinstr.at("out"));
      if (jinstr.contains("biases") && !jinstr.at("biases").is_null()) {
        n.biases = detail::parse_tid_obj(jinstr.at("biases"));
      } else {
        n.biases = std::nullopt;
      }
      n.group_size = jinstr.at("group_size").get<int>();
      n.bits       = jinstr.at("bits").get<int>();
      n.mode       = jinstr.at("mode").get<std::string>();
      n.out_dtype  = detail::parse_dtype(jinstr.at("out_dtype").get<std::string>());
      P.code.push_back(make_QUANTIZED_GATHER(std::move(n)));
    }
    else {
      throw std::runtime_error("program_from_json: unknown op " + op);
    }
  }

  return P;
}


inline ::mlx::core::Dtype to_mlx(DTypeId d) {
  using namespace ::mlx::core;
  switch (d) {
    case DTypeId::f16:    return float16;
    case DTypeId::f32:    return float32;
    case DTypeId::bf16:   return bfloat16;
    case DTypeId::i32:    return int32;
    case DTypeId::i64:    return int64;
    case DTypeId::u32:    return uint32;
    case DTypeId::u8:     return uint8;
    case DTypeId::boolean:return bool_;
  }
  throw std::runtime_error("to_mlx: unknown dtype");
}

// If you have a global/constexpr compute dtype, keep it here
static constexpr DTypeId kComputeDT = DTypeId::f32;


inline void bind_constants_from_safetensors(
    const std::string& path,
    Program& P,
    ConstantData& store)
{
  using namespace ::mlx::core;

  const uint32_t nconst = P.num_constant_tensors;
  if (nconst == 0) {
    store.tensors.clear();
    P.constants = &store;
    return;
  }

  // Load all tensors from safetensors
  // Adjust to your actual API
  auto tensors_pair = load_safetensors(path);
  const auto& tensors = tensors_pair.first;

  // collect all constant (tid,name) pairs from nameToSlot
  std::vector<std::pair<uint32_t, std::string>> const_slots;
  const_slots.reserve(P.nameToSlot.size());
  for (const auto& [name, slot] : P.nameToSlot) {
    if (std::holds_alternative<Tid>(slot)) {
      Tid t = std::get<Tid>(slot);
      if (t.idx < nconst) {
        const_slots.emplace_back(t.idx, name);
      }
    }
  }

  // we expect to have at least all constants named,
  // but to be robust, sort and check gaps
  std::sort(const_slots.begin(), const_slots.end(),
            [](auto& a, auto& b){ return a.first < b.first; });

  // rebuild constant area
  store.tensors.clear();
  store.tensors.reserve(nconst);

  auto to_mlx_dtype = [](DTypeId d){ return to_mlx(d); };

  auto to_target = [&](const array& ain,
                     DTypeId target_dt,
                     bool tpose = false) -> array {
    using namespace ::mlx::core;
    array a = ain;

    // if we need to transpose, do it first
    if (tpose && a.ndim() == 2)
      a = contiguous(transpose(a, {1, 0}));

    // for pure integer / packed types, DO NOT route through float32
    if (target_dt == DTypeId::u32 || target_dt == DTypeId::u8 || target_dt == DTypeId::i32) {
      if (a.dtype() != to_mlx(target_dt))
        a = astype(a, to_mlx(target_dt));
      return contiguous(a);
    }

    // existing float path
    if (a.dtype() != float32)
      a = astype(a, float32);
    a = contiguous(a);
    if (target_dt != DTypeId::f32)
      a = astype(a, to_mlx(target_dt));
    return a;
  };


  // now fill in order 0..nconst-1
  uint32_t next_expected = 0;
  for (const auto& [tid_idx, name] : const_slots) {
    if (tid_idx != next_expected) {
      // JSON said we have e.g. 10 constants but we only found names for some
      throw std::runtime_error(
        "bind_constants_from_safetensors: missing constant for Tid " +
        std::to_string(next_expected));
    }

    auto it = tensors.find(name);
    if (it == tensors.end()) {
      throw std::runtime_error("bind_constants_from_safetensors: missing key in safetensors: " + name);
    }

    // pick dtype from tensor_meta if available
    DTypeId target_dt = DTypeId::f32;
    if (tid_idx < P.tensor_meta.size() && P.tensor_meta[tid_idx].has_value()) {
      target_dt = P.tensor_meta[tid_idx]->dtype;
    }

    array cooked = to_target(it->second, target_dt, /*tpose=*/false);
    store.add(std::move(cooked));

    ++next_expected;
  }

  // if we didn’t cover all declared constants, complain
  if (next_expected != nconst) {
    throw std::runtime_error(
      "bind_constants_from_safetensors: program declares " +
      std::to_string(nconst) +
      " constant tensors, but only " +
      std::to_string(next_expected) +
      " were bound by name");
  }

  // hook it up
  P.bind_constants(store);
}

inline void init_execution_state_from_meta(const Program& P, ExecutionState& S) {
  if (S.P != &P)
    throw std::runtime_error("init_execution_state_from_meta: state not bound to this Program");

  using namespace ::mlx::core;

  const auto n_const = P.num_constant_tensors;
  const auto n_total = P.num_tensors();

  // tensor_meta can be smaller than total tensors
  for (uint32_t tidx = n_const; tidx < n_total; ++tidx) {
    const uint32_t slot = tidx - n_const;  // index into S.tensors
    if (tidx < P.tensor_meta.size() && P.tensor_meta[tidx].has_value()) {
      const auto& tm = *P.tensor_meta[tidx];

      // build MLX shape
      std::vector<int64_t> shape64;
      shape64.reserve(tm.shape.size());
      for (int d : tm.shape) {
        shape64.push_back(static_cast<int64_t>(d));
      }

      auto dtype = to_mlx(tm.dtype);

      // allocate zeros (you can switch to uninitialized if MLX exposes it)
      array a = zeros(::mlx::core::Shape(shape64.begin(), shape64.end()), dtype);
      S.tensors[slot] = std::move(a);
    } else {
      // leave as std::nullopt -> user/interpreter will fill it later
    }
  }
}


} // namespace executorch::mlx
