// program.hpp — refined Tid/Vid design with input/output maps
#pragma once
#include "ops.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <variant>
#include <unordered_map>
#include <type_traits>

#include <mlx/array.h>
#include <mlx/ops.h>

namespace executorch::mlx {

  // ============================================================================
// Core aliases
// ============================================================================
using Tensor = ::mlx::core::array;
using Value  = std::variant<int32_t, float, bool>;

// ============================================================================
// Tensor metadata (optional per tensor)
// ============================================================================
struct TensorMeta {
  std::vector<int> shape;
  std::vector<int> dim_order;
  DTypeId dtype;
};


// ============================================================================
// ConstantData — immutable storage for constants
// ============================================================================
struct ConstantData {
  std::vector<Tensor> tensors;

  inline const Tensor& const_tensor_ref(Tid id) const {
    if (id.idx >= tensors.size())
      throw std::out_of_range("ConstantData::const_tensor_ref: id out of range");
    return tensors[id.idx];
  }

  inline void add(Tensor t) { tensors.push_back(std::move(t)); }
};

// ============================================================================
// Program — immutable artifact (code + layout + meta)
// ============================================================================
struct Program {
  std::vector<Instr> code;

  // ---- Tensor layout ----
  uint32_t num_constant_tensors = 0;
  uint32_t num_non_constant_tensors = 0;
  uint32_t num_non_constant_values = 0;

  inline uint32_t num_tensors() const { return num_constant_tensors + num_non_constant_tensors; }
  inline uint32_t num_values() const { return num_non_constant_values;  }

  inline bool is_constant_tensor(Tid id) const { return id.idx < num_constant_tensors; }

  // ---- Tensor metadata ----
  std::vector<std::optional<TensorMeta>> tensor_meta;

  // ---- Constant data ----
  const ConstantData* constants = nullptr;

  // ---- Name → slot lookup ----
  using SlotVariant = std::variant<
      Tid,
      Vid<int32_t>,
      Vid<float>,
      Vid<bool>,
      Vid<std::string>>;

  std::unordered_map<std::string, SlotVariant> nameToSlot;

  // ---- Explicit I/O mappings ----
  // Each entry corresponds to user-facing positional index 0..N-1
  std::vector<SlotVariant> input_map;
  std::vector<SlotVariant> output_map;
  std::vector<SlotVariant> mutable_buffer_map;

  // ---- Bind constants ----
  inline void bind_constants(const ConstantData& data) {
    if (data.tensors.size() != num_constant_tensors)
      throw std::runtime_error("bind_constants: size mismatch");
    constants = &data;
  }

  // ---- Bind names ----
  inline void bind_name(Tid id, std::string name) {
    if (id.idx >= num_tensors())
      throw std::out_of_range("Program::bind_name: Tid out of range");
    if (!nameToSlot.emplace(std::move(name), id).second)
      throw std::runtime_error("Program::bind_name: duplicate name");
  }

  template <typename T>
  inline void bind_name(Vid<T> id, std::string name) {
    if (!nameToSlot.emplace(std::move(name), id).second)
      throw std::runtime_error("Program::bind_name: duplicate name");
  }

  // ---- Lookup ----
  inline SlotVariant get_slot(const std::string& name) const {
    auto it = nameToSlot.find(name);
    if (it == nameToSlot.end())
      throw std::runtime_error("Program::get_slot: unknown name '" + name + "'");
    return it->second;
  }

  // ---- I/O registration ----
  inline void add_input(SlotVariant s)  { input_map.push_back(std::move(s)); }
  inline void add_output(SlotVariant s) { output_map.push_back(std::move(s)); }
  inline void add_mutable_buffer(SlotVariant s) { mutable_buffer_map.push_back(std::move(s)); }
  
  inline const auto& input_slot(size_t i) const { return input_map.at(i); }
  inline const auto& output_slot(size_t i) const { return output_map.at(i); }

  inline size_t num_inputs() const  { return input_map.size(); }
  inline size_t num_outputs() const { return output_map.size(); }
};

// ============================================================================
// ExecutionState — per-run mutable data
// ============================================================================
struct ExecutionState {
  const Program* P = nullptr;

  std::vector<std::optional<Tensor>> tensors;
  std::vector<std::optional<Value>>  values;

  inline void bind(const Program& prog) {
    P = &prog;
    tensors.assign(P->num_non_constant_tensors, std::nullopt);
    values.assign(P->num_non_constant_values, std::nullopt);
  }

  // --------------------------
  // Tensor accessors
  // --------------------------
  inline Tensor& tensor_ref(Tid id) {
    if (!P) throw std::runtime_error("tensor_ref: Program not bound");
    if (id.idx >= P->num_tensors()) throw std::out_of_range("tensor_ref: id out of range");
    if (P->is_constant_tensor(id))
      throw std::runtime_error("tensor_ref: cannot mutate constant tensor");
    auto& opt = tensors[id.idx - P->num_constant_tensors];
    if (!opt) throw std::runtime_error("tensor_ref: uninitialized tensor idx=" + std::to_string(id.idx));
    return *opt;
  }

  inline const Tensor& const_tensor_ref(Tid id) const {
    if (!P) throw std::runtime_error("tensor_ref (const): Program not bound");
    if (id.idx >= P->num_tensors()) throw std::out_of_range("tensor_ref: id out of range");

    if (P->is_constant_tensor(id)) {
      if (!P->constants)
        throw std::runtime_error("tensor_ref (const): constants not bound");
      return P->constants->const_tensor_ref(id);
    }

    const auto& opt = tensors[id.idx - P->num_constant_tensors];
    if (!opt) throw std::runtime_error("tensor_ref (const): uninitialized tensor idx=" + std::to_string(id.idx));
    return *opt;
  }

  // --------------------------
  // Value accessors
  // --------------------------
  template <typename T>
  inline T& value_ref(Vid<T> id) {
    if (!P) throw std::runtime_error("value_ref: Program not bound");
    if (id.idx >= values.size()) throw std::out_of_range("value_ref: id out of range");
    auto& opt = values[id.idx];
    if (!opt) throw std::runtime_error("value_ref: uninitialized value idx=" + std::to_string(id.idx));
    return std::get<T>(*opt);
  }

  template <typename T>
  inline const T& const_value_ref(Vid<T> id) const {
    if (!P) throw std::runtime_error("value_ref (const): Program not bound");
    if (id.idx >= values.size()) throw std::out_of_range("value_ref: id out of range");
    const auto& opt = values[id.idx];
    if (!opt) throw std::runtime_error("value_ref (const): uninitialized value idx=" + std::to_string(id.idx));
    return std::get<T>(*opt);
  }

  // --------------------------
  // Name-based access
  // --------------------------
  inline Tensor& tensor_ref(const std::string& name) {
    auto slot = P->get_slot(name);
    return tensor_ref(std::get<Tid>(slot));
  }

  inline const Tensor& const_tensor_ref(const std::string& name) {
    auto slot = P->get_slot(name);
    return const_tensor_ref(std::get<Tid>(slot));
  }

  template <typename T>
  inline T& value_ref(const std::string& name) {
    auto slot = P->get_slot(name);
    return value_ref<T>(std::get<Vid<T>>(slot));
  }
};

} // namespace executorch::mlx
