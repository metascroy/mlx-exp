// program.hpp
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
#include <iostream> // @nocommit

namespace executorch::mlx {

// ---------------------------------------------
// Core aliases
// ---------------------------------------------
using Tensor = ::mlx::core::array;
using Scalar = std::variant<int32_t, float, bool>;   // only for non-constants

enum class SlotKind : uint8_t { Tensor, Scalar };

struct SlotMeta {
  SlotKind                   kind{SlotKind::Tensor};
  std::optional<std::string> name; // optional; empty = unnamed
};

// ---------------------------------------------
// Program: immutable artifact (code + constants + meta)
// ---------------------------------------------
struct Program {
  std::vector<Instr> code;

  // ---- Slot layout (contiguous ranges) ----
  uint32_t num_constants        = 0;  // tensors only
  uint32_t num_inputs           = 0;
  uint32_t num_outputs          = 0;
  uint32_t num_mutable_buffers  = 0;
  uint32_t num_temps            = 0;

  inline uint32_t constants_begin() const { return 0; }
  inline uint32_t inputs_begin()    const { return num_constants; }
  inline uint32_t outputs_begin()   const { return num_constants + num_inputs; }
  inline uint32_t mbufs_begin()     const { return num_constants + num_inputs + num_outputs; }
  inline uint32_t temps_begin()     const { return num_constants + num_inputs + num_outputs + num_mutable_buffers; }
  inline uint32_t total_slots()     const {
    return num_constants + num_inputs + num_outputs + num_mutable_buffers + num_temps;
  }

  // ---- Metadata: one entry per slot ----
  std::vector<SlotMeta> meta; // size == total_slots()

  // ---- Constants arena (read-only at runtime; tensors only) ----
  std::vector<std::optional<Tensor>> C_tensors;

  // Optional name -> Vid lookup
  std::unordered_map<std::string, Vid> nameToVid;

  // ----- Build / finalize helpers -----
  inline void finalize_layout() {
    const auto N = total_slots();
    meta.resize(N);
    C_tensors.resize(num_constants);
  }

  inline void bind_name(Vid id, std::string n) {
    if (id.idx >= meta.size()) throw std::runtime_error("Program::bind_name: id out of range");
    auto [it, ok] = nameToVid.emplace(n, id);
    if (!ok) throw std::runtime_error("Program::bind_name duplicate slot name: \"" + n + "\"");
    meta[id.idx].name = std::move(n);
  }

  inline Vid get_by_name(const std::string& name) const {
    auto it = nameToVid.find(name);
    if (it == nameToVid.end())
      throw std::runtime_error("Program::get_by_name: unknown name \"" + name + "\"");
    return it->second;
  }

  // Range-based role checks
  inline bool is_constant(Vid id) const { return id.idx < num_constants; }
  inline bool is_input  (Vid id) const { return id.idx >= inputs_begin()  && id.idx < outputs_begin(); }
  inline bool is_output (Vid id) const { return id.idx >= outputs_begin() && id.idx < mbufs_begin(); }
  inline bool is_mbuf   (Vid id) const { return id.idx >= mbufs_begin()   && id.idx < temps_begin(); }
  inline bool is_temp   (Vid id) const { return id.idx >= temps_begin()   && id.idx < total_slots(); }

  // Populate a constant tensor slot
  inline void set_constant_tensor(Vid id, Tensor t) {
    if (!is_constant(id)) throw std::runtime_error("set_constant_tensor: id not in constants range");
    if (meta[id.idx].kind != SlotKind::Tensor)
      throw std::runtime_error("set_constant_tensor: slot kind must be Tensor");
    C_tensors[id.idx] = std::move(t);
  }
};

// ---------------------------------------------
// ExecutionState: per-run mutable arena only
// ---------------------------------------------
struct ExecutionState {
  const Program* P = nullptr; // must be bound

  std::vector<std::optional<Tensor>> tensors; // size == P->total_slots()
  std::vector<std::optional<Scalar>> scalars; // size == P->total_slots()

  inline void bind(const Program& prog) {
    P = &prog;
    const auto N = P->total_slots();
    tensors.assign(N, std::nullopt);
    scalars.assign(N, std::nullopt);
  }

  // --------------------------
  // Tensor accessors
  // --------------------------
  inline Tensor& tensor_ref(Vid id) {
    if (!P) throw std::runtime_error("ExecutionState::tensor_ref (mut): Program not bound");
    if (id.idx >= P->total_slots()) throw std::out_of_range("tensor_ref: id out of range");
    if (P->meta[id.idx].kind != SlotKind::Tensor)
      throw std::runtime_error("tensor_ref: slot is not a Tensor");
    if (P->is_constant(id)) {
      throw std::runtime_error("tensor_ref (mut): write to constant Tensor slot");
    }

    auto& opt = tensors[id.idx];
    if (!opt) throw std::runtime_error("tensor_ref (mut): uninitialized Tensor idx=" + std::to_string(id.idx));
    return *opt;
  }

  inline const Tensor& tensor_ref(Vid id) const {
    if (!P) throw std::runtime_error("ExecutionState::tensor_ref (const): Program not bound");
    if (id.idx >= P->total_slots()) throw std::out_of_range("tensor_ref: id out of range");
    if (P->meta[id.idx].kind != SlotKind::Tensor)
      throw std::runtime_error("tensor_ref: slot is not a Tensor");

    if (P->is_constant(id)) {
      const auto& opt = P->C_tensors[id.idx];
      if (!opt) throw std::runtime_error("tensor_ref (const): missing constant Tensor idx=" + std::to_string(id.idx));
      return *opt;
    } else {
      const auto& opt = tensors[id.idx];
      if (!opt) throw std::runtime_error("tensor_ref (const): uninitialized Tensor idx=" + std::to_string(id.idx));
      return *opt;
    }
  }

  // --------------------------
  // Scalar accessors
  // --------------------------
  template <class T>
  inline T& scalar_ref(Vid id) {
    static_assert(std::is_same_v<T,int32_t> || std::is_same_v<T,float> || std::is_same_v<T,bool>,
                  "scalar_ref<T>: supported types are int32_t, float, bool");
    if (!P) throw std::runtime_error("ExecutionState::scalar_ref (mut): Program not bound");
    if (id.idx >= P->total_slots()) throw std::out_of_range("scalar_ref: id out of range");
    if (P->meta[id.idx].kind != SlotKind::Scalar)
      throw std::runtime_error("scalar_ref: slot is not a Scalar");
    if (P->is_constant(id))
      throw std::runtime_error("scalar_ref (mut): constants cannot be Scalars");

    auto& opt = scalars[id.idx];
    if (!opt) throw std::runtime_error("scalar_ref (mut): uninitialized Scalar idx=" + std::to_string(id.idx));
    return std::get<T>(*opt);
  }

  template <class T>
  inline const T& scalar_ref(Vid id) const {
    static_assert(std::is_same_v<T,int32_t> || std::is_same_v<T,float> || std::is_same_v<T,bool>,
                  "scalar_ref<T>: supported types are int32_t, float, bool");
    if (!P) throw std::runtime_error("ExecutionState::scalar_ref (const): Program not bound");
    if (id.idx >= P->total_slots()) throw std::out_of_range("scalar_ref: id out of range");
    if (P->meta[id.idx].kind != SlotKind::Scalar)
      throw std::runtime_error("scalar_ref: slot is not a Scalar");
    if (P->is_constant(id))
      throw std::runtime_error("scalar_ref (const): constants cannot be Scalars");

    const auto& opt = scalars[id.idx];
    if (!opt) throw std::runtime_error("scalar_ref (const): uninitialized Scalar idx=" + std::to_string(id.idx));
    return std::get<T>(*opt);
  }

  // --------------------------
  // By-name variants
  // --------------------------
  inline Tensor& tensor_ref(const std::string& name) {
    auto id = P->get_by_name(name);
    return tensor_ref(id);
  }
  inline const Tensor& tensor_ref(const std::string& name) const {
    auto id = P->get_by_name(name);
    return tensor_ref(id);
  }

  template <class T>
  inline T& scalar_ref(const std::string& name) {
    auto id = P->get_by_name(name);
    return scalar_ref<T>(id);
  }
  template <class T>
  inline const T& scalar_ref(const std::string& name) const {
    auto id = P->get_by_name(name);
    return scalar_ref<T>(id);
  }
};

} // namespace executorch::mlx
