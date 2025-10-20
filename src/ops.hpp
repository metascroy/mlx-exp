// ops.hpp
#pragma once
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <type_traits>
#include <utility>

// ----------------------------------------------------------------------------
// DType identifier shared with the interpreter (no magic ints).
// ----------------------------------------------------------------------------
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

using Tid = uint32_t;

// ----------------------------------------------------------------------------
// Op codes
// ----------------------------------------------------------------------------
enum class OpCode : uint8_t {
  NOOP = 0,            // default / placeholder

  // Math / linear
  MATMUL,
  MATMUL_ADD,
  RMS_NORM,

  // Attention
  SDPA,
  ROPE_APPLY,

  // Elementwise
  ADD, MUL, SILU,

  // Shapes
  RESHAPE, TRANSPOSE, CONTIGUOUS,

  // Indexing / cache
  GATHER, SLICE, CONCAT,

  // Dtype / const
  CAST, FULL, ZEROS, ONES,

  // Sampling
  ARGMAX,
};

// ----------------------------------------------------------------------------
// Per-op payloads
// ----------------------------------------------------------------------------

struct NoopNode { };  // placeholder / default payload

struct MatmulNode {
  Tid a{}, b{}, out{};
  bool ta{false}, tb{false};
  std::optional<Tid> bias{};
};

struct RMSNormNode {
  Tid x{}, weight{}, out{};
  float eps{1e-5f};
};

struct RopeNode {
  Tid q_in{}, k_in{}, cos_tbl{}, sin_tbl{};
  Tid q_out{}, k_out{};
  int head_dim{};
  int pos_offset{0};
};

struct SdpaNode {
  Tid q{}, k{}, v{}, out{};
  float scale{1.0f};
  std::optional<Tid> mask{};
};

struct AddNode    { Tid a{}, b{}, out{}; };
struct MulNode    { Tid a{}, b{}, out{}; };
struct SiluNode   { Tid x{}, out{}; };

struct ReshapeNode   { Tid x{}, out{}; std::vector<int> shape; };
struct TransposeNode { Tid x{}, out{}; std::vector<int> perm;  };
struct ContigNode    { Tid x{}, out{}; };

struct GatherNode { Tid table{}, ids{}, out{}; };
struct SliceNode  {
  Tid x{}, out{};
  int axis{}, start{}, end{};

};
struct ConcatNode { Tid a{}, b{}, out{}; int axis{}; };

struct CastNode   { Tid x{}, out{}; DTypeId dtype{DTypeId::f16}; };
struct FullNode   { Tid out{}; std::vector<int> shape; float v{}; DTypeId dtype{DTypeId::f16}; };
struct ZerosNode  { Tid out{}; std::vector<int> shape;            DTypeId dtype{DTypeId::f16}; };
struct OnesNode   { Tid out{};  std::vector<int> shape;           DTypeId dtype{DTypeId::f16}; };

struct ArgmaxNode { Tid x{}, out{}; int axis{}; };

// ----------------------------------------------------------------------------
// Variant payload + thin instruction wrapper
// ----------------------------------------------------------------------------

using NodeVariant = std::variant<
  NoopNode,
  MatmulNode,
  RMSNormNode,
  RopeNode,
  SdpaNode,
  AddNode,
  MulNode,
  SiluNode,
  ReshapeNode,
  TransposeNode,
  ContigNode,
  GatherNode,
  SliceNode,
  ConcatNode,
  CastNode,
  FullNode,
  ZerosNode,
  OnesNode,
  ArgmaxNode
>;

struct Instr {
  OpCode      op{OpCode::NOOP};
  NodeVariant node{NoopNode{}}; // safe default

  Instr() = default;

  template <class T,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<T>, Instr>>>
  Instr(OpCode opcode, T&& payload)
  : op(opcode), node(std::forward<T>(payload)) {}

  template <class T>       T& get()       { return std::get<T>(node); }
  template <class T> const T& get() const { return std::get<T>(node); }

  template <class F> decltype(auto) visit(F&& f)       { return std::visit(std::forward<F>(f), node); }
  template <class F> decltype(auto) visit(F&& f) const { return std::visit(std::forward<F>(f), node); }
};

// ----------------------------------------------------------------------------
// Small convenience factories
// ----------------------------------------------------------------------------
inline Instr make_noop()                   { return Instr{OpCode::NOOP,       NoopNode{}}; }
inline Instr make_matmul(MatmulNode n)     { return Instr{OpCode::MATMUL,     std::move(n)}; }
inline Instr make_matmul_add(MatmulNode n) { return Instr{OpCode::MATMUL_ADD, std::move(n)}; }
inline Instr make_rmsnorm(RMSNormNode n)   { return Instr{OpCode::RMS_NORM,   std::move(n)}; }
inline Instr make_sdpa(SdpaNode n)         { return Instr{OpCode::SDPA,       std::move(n)}; }
inline Instr make_rope(RopeNode n)         { return Instr{OpCode::ROPE_APPLY, std::move(n)}; }

inline Instr make_add(AddNode n)           { return Instr{OpCode::ADD,        std::move(n)}; }
inline Instr make_mul(MulNode n)           { return Instr{OpCode::MUL,        std::move(n)}; }
inline Instr make_silu(SiluNode n)         { return Instr{OpCode::SILU,       std::move(n)}; }

inline Instr make_reshape(ReshapeNode n)   { return Instr{OpCode::RESHAPE,    std::move(n)}; }
inline Instr make_transpose(TransposeNode n){return Instr{OpCode::TRANSPOSE,  std::move(n)}; }
inline Instr make_contig(ContigNode n)     { return Instr{OpCode::CONTIGUOUS, std::move(n)}; }

inline Instr make_gather(GatherNode n)     { return Instr{OpCode::GATHER,     std::move(n)}; }
inline Instr make_slice(SliceNode n)       { return Instr{OpCode::SLICE,      std::move(n)}; }
inline Instr make_concat(ConcatNode n)     { return Instr{OpCode::CONCAT,     std::move(n)}; }

inline Instr make_cast(CastNode n)         { return Instr{OpCode::CAST,       std::move(n)}; }
inline Instr make_full(FullNode n)         { return Instr{OpCode::FULL,       std::move(n)}; }
inline Instr make_zeros(ZerosNode n)       { return Instr{OpCode::ZEROS,      std::move(n)}; }
inline Instr make_ones(OnesNode n)         { return Instr{OpCode::ONES,       std::move(n)}; }

inline Instr make_argmax(ArgmaxNode n)     { return Instr{OpCode::ARGMAX,     std::move(n)}; }
