


#pragma once
#include <cstdint>
#include <variant>

// ----------------------------------------
// Strongly-typed IDs
// ----------------------------------------

// Mutable tensor ID
struct Mid {
  uint32_t idx{};
  constexpr explicit Mid(uint32_t i=0) : idx(i) {}
};

// Constant tensor ID
struct Cid {
  uint32_t idx{};
  constexpr explicit Cid(uint32_t i=0) : idx(i) {}
};

// Tensor ID
using Tid = std::variant<Mid, Cid>;

// Mutable param IDs
struct I32Id   { uint32_t idx{}; constexpr explicit I32Id(uint32_t i=0):idx(i){} };
struct F32Id   { uint32_t idx{}; constexpr explicit F32Id(uint32_t i=0):idx(i){} };
struct ShapeId { uint32_t idx{}; constexpr explicit ShapeId(uint32_t i=0):idx(i){} };

enum class DTypeId : int {
  f16 = 0,
  f32 = 1,
  bf16 = 2,
  i32 = 3,
  i64 = 4,
  u32 = 5,
  u8  = 6,
  boolean = 7,
};
