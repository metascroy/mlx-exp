// ops.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <optional>
#include <variant>
#include <vector>
#include <utility>
#include <string>  // for Quantized*::mode

// Unified value identifier. Role is inferred from its index range in Program.
struct Vid { uint32_t idx{}; };

// DType tag for node attributes (not MLX dtype proper; map in executor)
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

// -----------------------------------------------------------------------------
// Per-op payloads (schemas) — all inputs/outputs/constants are Vid
// -----------------------------------------------------------------------------
struct NoopNode { };

// Linear (no implicit transposes; supply weights in the shape you want)
struct LinearNode {
  Vid x{}, weight{};
  Vid out{};
  std::optional<Vid> bias{};  // neural bias (post-matmul), usually constant
};

struct RMSNormNode {
  Vid x{}, weight{};
  Vid out{};
  float eps{1e-5f};
};

struct RopeNode {
  // inputs
  Vid q_in{}, k_in{};

  // optional precomputed frequency spectrum (inv_freq). If absent, kernel derives from base/dims.
  std::optional<Vid> freq{std::nullopt};  // usually a constant tensor

  // outputs
  Vid q_out{}, k_out{};

  // params
  int  head_dim{0};
  bool traditional{false};
  std::optional<float> base{500000.f};
  float scale{1.0f};

  Vid pos{};  // runtime scalar (int32) cursor
};

struct SdpaNode {
  Vid q{}, k{}, v{};
  Vid out{};
  float scale{1.0f};
  std::optional<Vid> mask{};  // optional additive or boolean mask tensor
  bool causal{false};         // if true and mask unset, execute causal attention
};

struct AddNode  { Vid a{}, b{}; Vid out{}; };
struct MulNode  { Vid a{}, b{}; Vid out{}; };
struct SiluNode { Vid x{};      Vid out{}; };

struct ReshapeNode   { Vid x{}; Vid out{}; std::vector<int> shape; };
struct TransposeNode { Vid x{}; Vid out{}; std::vector<int> perm;  };
struct ContigNode    { Vid x{}; Vid out{}; };

struct GatherNode { Vid table{}, ids{}; Vid out{}; };

/**
 * Slice (single axis):
 *  Selects a contiguous segment along `axis`.
 *    - axis:  Vid of int scalar (target axis in x)
 *    - start: Vid of int scalar (inclusive)
 *    - length: Vid of int scalar (#elements; if negative, treat as "to end")
 */
struct SliceNode {
  Vid x{};
  Vid out{};
  Vid axis{};    // int scalar
  Vid start{};   // int scalar
  Vid length{};  // int scalar; negative => to end
};

struct ConcatNode { Vid a{}, b{}; Vid out{}; int axis{0}; };

struct CastNode  { Vid x{};  Vid out{}; DTypeId dtype{DTypeId::f16}; };
struct FullNode  {             Vid out{}; std::vector<int> shape; float v{0.0f}; DTypeId dtype{DTypeId::f16}; };
struct ZerosNode {             Vid out{}; std::vector<int> shape;                  DTypeId dtype{DTypeId::f16}; };
struct OnesNode  {             Vid out{}; std::vector<int> shape;                  DTypeId dtype{DTypeId::f16}; };

struct ArgmaxNode { Vid x{}; Vid out{}; int axis{-1}; };

/**
 * SliceUpdate (single axis, in-place on dst):
 *  Overwrites a contiguous segment of `dst` along `axis` starting at `start`
 *  with `update`. If `length` is negative, infer from update.size(axis).
 */
struct SliceUpdateNode {
  Vid dst{};
  Vid update{};
  Vid axis{};    // int scalar
  Vid start{};   // int scalar
  Vid length{};  // int scalar; negative => infer from update.shape(axis)
};

// Quantized linear using MLX fast quantized paths
struct QuantizedLinearNode {
  Vid x{};                         // [B, I] input (f16/f32/bf16)
  Vid w{};                         // quantized weight (uint8) — constant
  Vid scales{};                    // per-group scales — constant
  std::optional<Vid> biases{};     // per-group *quantization* biases (affine dequant), optional
  std::optional<Vid> bias{};       // neural bias vector (post-matmul), optional
  int  group_size{64};             // e.g., 64
  int  bits{4};                    // number of quantization bits (e.g., 4)
  std::string mode{"affine"};      // "affine" or "symmetric"
  DTypeId out_dtype{DTypeId::f32}; // f16/f32/bf16
  Vid out{};                       // output
};

// Quantized embedding/gather with dequantization (no neural bias here)
struct QuantizedGatherNode {
  Vid table_q{};                   // quantized embedding table [vocab, Dm] — constant (uint8 nibbles)
  Vid scales{};                    // per-group scales — constant
  std::optional<Vid> biases{};     // per-group *quantization* biases (affine dequant), optional
  int  group_size{64};             // quantization group size along the feature dim
  int  bits{4};                    // number of quantization bits (e.g., 4)
  std::string mode{"affine"};      // "affine" or "symmetric"
  DTypeId out_dtype{DTypeId::f32}; // output activation dtype
  Vid ids{};                       // [B, T] token ids
  Vid out{};                       // [B, T, Dm] dequantized embedding activations
};

// -----------------------------------------------------------------------------
// X-macro master list: single source of truth (NAME, PAYLOAD_TYPE)
// -----------------------------------------------------------------------------
#ifndef LLM_OP_LIST
#define LLM_OP_LIST(X)                            \
  X(NOOP,                NoopNode)                \
  /* Math / linear */                             \
  X(LINEAR,              LinearNode)              \
  X(RMS_NORM,            RMSNormNode)             \
  /* Attention */                                 \
  X(SDPA,                SdpaNode)                \
  X(ROPE_APPLY,          RopeNode)                \
  /* Elementwise */                               \
  X(ADD,                 AddNode)                 \
  X(MUL,                 MulNode)                 \
  X(SILU,                SiluNode)                \
  /* Shapes */                                    \
  X(RESHAPE,             ReshapeNode)             \
  X(TRANSPOSE,           TransposeNode)           \
  X(CONTIGUOUS,          ContigNode)              \
  /* Indexing / cache */                          \
  X(GATHER,              GatherNode)              \
  X(SLICE,               SliceNode)               \
  X(CONCAT,              ConcatNode)              \
  /* Dtype / const */                             \
  X(CAST,                CastNode)                \
  X(FULL,                FullNode)                \
  X(ZEROS,               ZerosNode)               \
  X(ONES,                OnesNode)                \
  /* Sampling */                                  \
  X(ARGMAX,              ArgmaxNode)              \
  /* In-place */                                  \
  X(SLICE_UPDATE,        SliceUpdateNode)         \
  /* Quantized */                                  \
  X(QUANTIZED_GATHER,    QuantizedGatherNode)     \
  X(QUANTIZED_LINEAR,    QuantizedLinearNode)
#endif

// -----------------------------------------------------------------------------
// OpCode enum (contiguous) + sentinel
// -----------------------------------------------------------------------------
enum class OpCode : uint8_t {
#define DEFINE_ENUM(NAME, PAYLOAD) NAME,
  LLM_OP_LIST(DEFINE_ENUM)
#undef DEFINE_ENUM
  SENTINEL
};

// -----------------------------------------------------------------------------
// Traits: OpCode -> Payload type
// -----------------------------------------------------------------------------
template <OpCode> struct OpPayload;
#define DEFINE_TRAIT(NAME, PAYLOAD) \
  template <> struct OpPayload<OpCode::NAME> { using type = PAYLOAD; };
LLM_OP_LIST(DEFINE_TRAIT)
#undef DEFINE_TRAIT

template <OpCode OC>
using OpPayloadT = typename OpPayload<OC>::type;

// -----------------------------------------------------------------------------
// NodeVariant (allows duplicate payload types; use index-based emplace)
// -----------------------------------------------------------------------------
using NodeVariant = std::variant<
#define VAR_ALT(NAME, PAYLOAD) PAYLOAD,
  LLM_OP_LIST(VAR_ALT)
#undef VAR_ALT
  std::monostate
>;

// Generate stable indices for each opcode’s alternative in NodeVariant
enum : size_t {
#define ENUM_IDX(NAME, PAYLOAD) VAR_IDX_##NAME,
  LLM_OP_LIST(ENUM_IDX)
#undef ENUM_IDX
  VAR_IDX_SENTINEL
};

template <OpCode> struct OpVariantIndex;
#define DEFINE_INDEX_TRAIT(NAME, PAYLOAD) \
  template <> struct OpVariantIndex<OpCode::NAME> { static constexpr size_t value = VAR_IDX_##NAME; };
LLM_OP_LIST(DEFINE_INDEX_TRAIT)
#undef DEFINE_INDEX_TRAIT

static_assert(std::variant_size<NodeVariant>::value >= VAR_IDX_SENTINEL,
              "NodeVariant must have at least as many alts as ops");

// Optional: names for debug/logging
static constexpr const char* kOpName[static_cast<size_t>(OpCode::SENTINEL)] = {
#define NAME_ROW(NAME, PAYLOAD) #NAME,
  LLM_OP_LIST(NAME_ROW)
#undef NAME_ROW
};
static_assert(sizeof(kOpName) / sizeof(kOpName[0]) ==
              static_cast<size_t>(OpCode::SENTINEL),
              "kOpName size must match OpCode::SENTINEL");

// -----------------------------------------------------------------------------
// Instruction (compile-time factory uses index-based emplace to disambiguate)
// -----------------------------------------------------------------------------
struct Instr {
  OpCode      op{OpCode::NOOP};
  NodeVariant node{NoopNode{}};

  Instr() = default;

  template <OpCode OC>
  static Instr make(OpPayloadT<OC> payload) {
    Instr i;
    i.op = OC;
    // Emplace by index so duplicate types are unambiguous
    i.node.template emplace<OpVariantIndex<OC>::value>(std::move(payload));
    return i;
  }

  template <class T>       T& get()       { return std::get<T>(node); }
  template <class T> const T& get() const { return std::get<T>(node); }

  template <class F> decltype(auto) visit(F&& f)       { return std::visit(std::forward<F>(f), node); }
  template <class F> decltype(auto) visit(F&& f) const { return std::visit(std::forward<F>(f), node); }
};

// Sanity: OpCode::COUNT matches LLM_OP_LIST item count
static_assert(static_cast<size_t>(OpCode::SENTINEL) == ([]{
  size_t n = 0;
#define COUNT_ONE(NAME, PAYLOAD) ++n;
  LLM_OP_LIST(COUNT_ONE)
#undef COUNT_ONE
  return n;
})(), "OpCode::COUNT mismatch with LLM_OP_LIST");

// -----------------------------------------------------------------------------
// Auto-generated convenience factories (no drift — built from LLM_OP_LIST)
// -----------------------------------------------------------------------------
#define DEFINE_MAKE_FN(NAME, PAYLOAD) \
  inline Instr make_##NAME(PAYLOAD n) { \
    return Instr::make<OpCode::NAME>(std::move(n)); \
  }
LLM_OP_LIST(DEFINE_MAKE_FN)
#undef DEFINE_MAKE_FN
