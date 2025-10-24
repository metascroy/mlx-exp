// program.hpp
#pragma once
#include "ops.hpp"   // Mid, Cid, Tid, I32Id, F32Id, ShapeId, Instr

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <variant>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace  executorch::mlx {

// ----------------------------------------------------------------------------
// Aliases / helpers
// ----------------------------------------------------------------------------
using Tensor = ::mlx::core::array;

// ----------------------------------------------------------------------------
// Immutable constants (treat as read-only after build)
// ----------------------------------------------------------------------------
struct ConstantData {
  std::vector<Tensor> C;                          // weights, tables, fixed masks

  inline const Tensor& c_ref(Cid id) const { return C.at(id.idx); }

  inline Cid add(Tensor t) {
    C.emplace_back(std::move(t));
    // keepalive.emplace_back(nullptr);
    return Cid{static_cast<uint32_t>(C.size() - 1)};
  }
};

// ----------------------------------------------------------------------------
// Mutable per-session data (inputs, outputs, state, temps, scalars, shapes)
// ----------------------------------------------------------------------------
struct MutableData {
  using Tensor = ::mlx::core::array;

  // Sparse storage keyed by ID indices
  std::unordered_map<uint32_t, Tensor>              M;      // Mid -> tensor
  std::unordered_map<uint32_t, int>                 i32;    // I32Id -> value
  std::unordered_map<uint32_t, float>               f32;    // F32Id -> value
  std::unordered_map<uint32_t, ::mlx::core::Shape>  shape;  // ShapeId -> shape

  // Optionally reserve buckets up-front (helps perf if you know rough counts)
  inline void reserve(size_t num_tensors,
                      size_t num_i32 = 8,
                      size_t num_f32 = 8,
                      size_t num_shapes = 8) {
    M.reserve(num_tensors);
    i32.reserve(num_i32);
    f32.reserve(num_f32);
    shape.reserve(num_shapes);
  }

  // ---- setters (no pre-size; just insert/overwrite) ----
  inline void set_mutable_id(Mid id, Tensor v) {
    M.insert_or_assign(id.idx, std::move(v));
  }
  inline void set_i32_id(I32Id id, int v) {
    i32[id.idx] = v;
  }
  inline void set_f32_id(F32Id id, float v) {
    f32[id.idx] = v;
  }
  inline void set_shape_id(ShapeId id, const std::vector<int>& v) {
    shape[id.idx] = ::mlx::core::Shape(v.begin(), v.end());
  }

  // ---- refs (throwing if missing) ----
  inline Tensor& m_ref(Mid id) {
    auto it = M.find(id.idx);
    if (it == M.end())
      throw std::runtime_error("Missing tensor for Mid idx=" + std::to_string(id.idx));
    return it->second;
  }
  inline const Tensor& m_ref(Mid id) const {
    auto it = M.find(id.idx);
    if (it == M.end())
      throw std::runtime_error("Missing tensor for Mid idx=" + std::to_string(id.idx));
    return it->second;
  }

  inline int& i32_ref(I32Id id) {
    auto it = i32.find(id.idx);
    if (it == i32.end())
      throw std::out_of_range("i32_ref: uninitialized I32Id idx=" + std::to_string(id.idx));
    return it->second;
  }
  inline const int& i32_ref(I32Id id) const {
    auto it = i32.find(id.idx);
    if (it == i32.end())
      throw std::out_of_range("i32_ref(const): uninitialized I32Id idx=" + std::to_string(id.idx));
    return it->second;
  }

  inline float& f32_ref(F32Id id) {
    auto it = f32.find(id.idx);
    if (it == f32.end())
      throw std::out_of_range("f32_ref: uninitialized F32Id idx=" + std::to_string(id.idx));
    return it->second;
  }
  inline const float& f32_ref(F32Id id) const {
    auto it = f32.find(id.idx);
    if (it == f32.end())
      throw std::out_of_range("f32_ref(const): uninitialized F32Id idx=" + std::to_string(id.idx));
    return it->second;
  }

  inline ::mlx::core::Shape& shape_ref(ShapeId id) {
    auto it = shape.find(id.idx);
    if (it == shape.end())
      throw std::out_of_range("shape_ref: uninitialized ShapeId idx=" + std::to_string(id.idx));
    return it->second;
  }
  inline const ::mlx::core::Shape& shape_ref(ShapeId id) const {
    auto it = shape.find(id.idx);
    if (it == shape.end())
      throw std::out_of_range("shape_ref(const): uninitialized ShapeId idx=" + std::to_string(id.idx));
    return it->second;
  }
};

// ----------------------------------------------------------------------------
// Program: immutable artifact (code + constant tensors).
// ----------------------------------------------------------------------------
struct Program {
  std::vector<Instr>                 code;  // immutable after build
  std::shared_ptr<const ConstantData> C;    // must be set
};

// ----------------------------------------------------------------------------
// Unbound helpers (preferred for concurrency)
// ----------------------------------------------------------------------------
inline const Tensor& read_tensor(const Program& p,
                                 const MutableData& m,
                                 const Tid& t)
{
  if (std::holds_alternative<Mid>(t)) {
    return m.m_ref(std::get<Mid>(t));
  } else {
    if (!p.C) throw std::runtime_error("Program constants not set");
    return p.C->c_ref(std::get<Cid>(t));
  }
}

inline void write_tensor(MutableData& m, Mid id, Tensor v) {
  m.set_mutable_id(id, std::move(v));
}

// small scalar/shape helpers to mirror write_tensor (optional)
inline int   read_i32 (const MutableData& m, I32Id id) { return m.i32_ref(id); }
inline float read_f32 (const MutableData& m, F32Id id) { return m.f32_ref(id); }
inline const ::mlx::core::Shape& read_shape(const MutableData& m, ShapeId id) { return m.shape_ref(id); }

} // namespace executorch::mlx
