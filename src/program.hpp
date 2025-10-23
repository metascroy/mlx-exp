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
  std::vector<std::optional<Tensor>> M;
  std::vector<int>              i32;
  std::vector<float>            f32;
  std::vector<::mlx::core::Shape> shape;

  // ---- setters ----
  inline void set_mutable_id(Mid id, Tensor v) {
    if (id.idx >= M.size()) M.resize(id.idx + 1);
    M[id.idx] = std::move(v);
  }
  inline void set_i32_id(I32Id id, int v) {
    if (id.idx >= i32.size()) i32.resize(id.idx + 1);
    i32[id.idx] = v;
  }
  inline void set_f32_id(F32Id id, float v) {
    if (id.idx >= f32.size()) f32.resize(id.idx + 1);
    f32[id.idx] = v;
  }
  inline void set_shape_id(ShapeId id, const std::vector<int>& v) {
    if (id.idx >= shape.size()) shape.resize(id.idx + 1);
    shape[id.idx] = ::mlx::core::Shape(v.begin(), v.end());
  }

  // ---- refs (throwing) ----
  inline Tensor& m_ref(Mid id) {
    if (id.idx >= M.size() || !M[id.idx].has_value())
      throw std::runtime_error("Missing tensor for Mid idx=" + std::to_string(id.idx));
    return M[id.idx].value();
  }
  inline const Tensor& m_ref(Mid id) const {
    if (id.idx >= M.size() || !M[id.idx].has_value())
      throw std::runtime_error("Missing tensor for Mid idx=" + std::to_string(id.idx));
    return M[id.idx].value();
  }
  inline int& i32_ref(I32Id id) {
    if (id.idx >= i32.size())
      throw std::out_of_range("i32_ref: uninitialized I32Id idx=" + std::to_string(id.idx));
    return i32[id.idx];
  }
  inline float& f32_ref(F32Id id) {
    if (id.idx >= f32.size())
      throw std::out_of_range("f32_ref: uninitialized F32Id idx=" + std::to_string(id.idx));
    return f32[id.idx];
  }
  inline ::mlx::core::Shape& shape_ref(ShapeId id) {
    if (id.idx >= shape.size())
      throw std::out_of_range("shape_ref: uninitialized ShapeId idx=" + std::to_string(id.idx));
    return shape[id.idx];
  }
  inline const int& i32_ref(I32Id id) const {
    if (id.idx >= i32.size())
      throw std::out_of_range("i32_ref(const): uninitialized I32Id idx=" + std::to_string(id.idx));
    return i32[id.idx];
  }
  inline const float& f32_ref(F32Id id) const {
    if (id.idx >= f32.size())
      throw std::out_of_range("f32_ref(const): uninitialized F32Id idx=" + std::to_string(id.idx));
    return f32[id.idx];
  }
  inline const ::mlx::core::Shape& shape_ref(ShapeId id) const {
    if (id.idx >= shape.size())
      throw std::out_of_range("shape_ref(const): uninitialized ShapeId idx=" + std::to_string(id.idx));
    return shape[id.idx];
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
