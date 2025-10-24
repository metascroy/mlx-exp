// ops.hpp
#pragma once
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <type_traits>
#include <utility>
#include <functional>
#include <array>
#include <string>     // ← needed for QLinear4Node::mode / QEmbed4Node::mode
#include "id.hpp"     // Mid, Cid, Tid=std::variant<Mid,Cid>, I32Id, F32Id, ShapeId, DTypeId

// -----------------------------------------------------------------------------
// Per-op payloads (schemas)
// -----------------------------------------------------------------------------
struct NoopNode { };

struct MatmulNode {
  Tid a{}, b{};
  Mid out{};
  bool ta{false}, tb{false};
  std::optional<Tid> bias{};
};

struct RMSNormNode {
  Tid x{}, weight{};
  Mid out{};
  float eps{1e-5f};
};

struct RopeNode {
  // inputs
  Tid q_in{}, k_in{};

  // optional precomputed frequency spectrum (a.k.a. inv_freq). If absent, kernel derives from base/dims.
  std::optional<Tid> freq{std::nullopt};

  // outputs
  Mid q_out{}, k_out{};

  // params
  int  head_dim{0};
  bool traditional{false};
  std::optional<float> base{500000.f};
  float scale{1.0f};

  I32Id pos{}; // runtime offset/cursor
};

struct SdpaNode {
  Tid q{}, k{}, v{};
  Mid out{};
  float scale{1.0f};
  std::optional<Tid> mask{};  // optional additive or boolean mask
  bool causal{false};         // pass "causal" string mask to MLX if true and no tensor mask is provided
};

struct AddNode  { Tid a{}, b{}; Mid out{}; };
struct MulNode  { Tid a{}, b{}; Mid out{}; };
struct SiluNode { Tid x{};      Mid out{}; };

struct ReshapeNode   { Tid x{}; Mid out{}; std::vector<int> shape; };
struct TransposeNode { Tid x{}; Mid out{}; std::vector<int> perm;  };
struct ContigNode    { Tid x{}; Mid out{}; };

struct GatherNode { Tid table{}, ids{}; Mid out{}; };

struct SliceNode {
  Tid x{};
  Mid out{};
  ShapeId start{};
  ShapeId stop{};                    // -1 => to end of that dim (handled at runtime)
  std::optional<ShapeId> strides{};  // nullopt => contiguous (all 1s)
};

struct ConcatNode { Tid a{}, b{}; Mid out{}; int axis{0}; };

struct CastNode  { Tid x{};  Mid out{}; DTypeId dtype{DTypeId::f16}; };
struct FullNode  {            Mid out{}; std::vector<int> shape; float v{0.0f}; DTypeId dtype{DTypeId::f16}; };
struct ZerosNode {            Mid out{}; std::vector<int> shape;                  DTypeId dtype{DTypeId::f16}; };
struct OnesNode  {            Mid out{}; std::vector<int> shape;                  DTypeId dtype{DTypeId::f16}; };

struct ArgmaxNode { Tid x{}; Mid out{}; int axis{-1}; };

struct SliceUpdateNode {
  Mid dst{};
  Tid update{};
  ShapeId start{};
  ShapeId stop{};                     // -1 => infer = start + update.shape()
  std::optional<ShapeId> strides{};   // nullopt / all 1s => contiguous
};

// Quantized 4-bit linear using MLX fast::quantized_matmul
struct QLinear4Node {
  Tid x{};                 // [B, I] input (f16/f32/bf16)
  Cid w{};                 // quantized weight (uint8)
  Cid scales{};            // per-group scales
  std::optional<Cid> biases{}; // optional "biases" from quantize() / for dequantize()
  bool transpose{true};    // usually true (W^T multiply)
  int group_size{64};      // e.g., 64
  std::string mode{"affine"}; // "affine" or "symmetric"
  DTypeId out_dtype{DTypeId::f32}; // f16/f32/bf16
  Mid out{};               // output (mutable)
};

// Quantized 4-bit embedding lookup: gather rows from a Q4 table and dequantize
struct QEmbed4Node {
  Cid table_q4{};              // quantized embedding table [vocab, Dm] packed in uint8 nibbles
  Cid scales{};                // per-group scales
  std::optional<Cid> biases{}; // optional "biases" from quantize() (used by dequantize)
  int group_size{64};          // quantization group size along the feature dim
  std::string mode{"affine"};  // "affine" or "symmetric"
  DTypeId out_dtype{DTypeId::f32}; // output activation dtype
  Tid ids{};                   // [B, T] token ids
  Mid out{};                   // [B, T, Dm] dequantized embedding activations
};

// -----------------------------------------------------------------------------
// X-macro master list: single source of truth (NAME, PAYLOAD_TYPE)
// -----------------------------------------------------------------------------
#ifndef LLM_OP_LIST
#define LLM_OP_LIST(X)                      \
  X(NOOP,          NoopNode)                \
  /* Math / linear */                       \
  X(MATMUL,        MatmulNode)              \
  X(RMS_NORM,      RMSNormNode)             \
  /* Attention */                           \
  X(SDPA,          SdpaNode)                \
  X(ROPE_APPLY,    RopeNode)                \
  /* Elementwise */                         \
  X(ADD,           AddNode)                 \
  X(MUL,           MulNode)                 \
  X(SILU,          SiluNode)                \
  /* Shapes */                              \
  X(RESHAPE,       ReshapeNode)             \
  X(TRANSPOSE,     TransposeNode)           \
  X(CONTIGUOUS,    ContigNode)              \
  /* Indexing / cache */                    \
  X(GATHER,        GatherNode)              \
  X(SLICE,         SliceNode)               \
  X(CONCAT,        ConcatNode)              \
  /* Dtype / const */                       \
  X(CAST,          CastNode)                \
  X(FULL,          FullNode)                \
  X(ZEROS,         ZerosNode)               \
  X(ONES,          OnesNode)                \
  /* Sampling */                            \
  X(ARGMAX,        ArgmaxNode)              \
  /* In-place */                            \
  X(SLICE_UPDATE,  SliceUpdateNode)         \
  /* Quantized */                           \
  X(QEMBED4,       QEmbed4Node)             \
  X(QLINEAR4,      QLinear4Node)
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
