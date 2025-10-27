// interpreter.hpp
#pragma once
#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ops.hpp"
#include "program.hpp"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <iostream> // @nocommit

namespace executorch::mlx {

// -----------------------------------------------------------------------------
// Local helpers
// -----------------------------------------------------------------------------
static inline ::mlx::core::Shape to_shape(const std::vector<int>& v) {
  return ::mlx::core::Shape(v.begin(), v.end());
}

static inline ::mlx::core::Dtype to_dtype(DTypeId d) {
  using namespace ::mlx::core;
  switch (d) {
    case DTypeId::f16:     return float16;
    case DTypeId::f32:     return float32;
    case DTypeId::bf16:    return bfloat16;
    case DTypeId::i32:     return int32;
    case DTypeId::i64:     return int64;
    case DTypeId::u32:     return uint32;
    case DTypeId::u8:      return uint8;
    case DTypeId::boolean: return bool_;
  }
  return ::mlx::core::float32;
}

// New: uniform way to write op outputs into ExecutionState
static inline void set_output(ExecutionState& st, Tid out, ::mlx::core::array a) {
  if (!st.P) throw std::runtime_error("set_output: Program not bound");
  if (out.idx < st.P->num_constant_tensors)
    throw std::runtime_error("set_output: cannot write to constant tensor");
  const uint32_t off = out.idx - st.P->num_constant_tensors;
  if (off >= st.tensors.size()) throw std::out_of_range("set_output: tensor idx out of range");
  st.tensors[off] = std::move(a);
}

// Alias for brevity
using StreamOrDevice = ::mlx::core::StreamOrDevice;

// -----------------------------------------------------------------------------
// Heavy op implementations (stream-aware)
// -----------------------------------------------------------------------------
namespace impl {

// ----- Linear (no implicit transposes) -----
inline void do_linear(const Program&,
                      ExecutionState& state,
                      const LinearNode& n,
                      StreamOrDevice s) {
  using namespace ::mlx::core;

  const array& X = state.const_tensor_ref(n.x);
  const array& W = state.const_tensor_ref(n.weight);

  auto WT = transpose(W, {1, 0}, s);
  array Y = matmul(X, WT, s);  // assume shapes already correct

  if (n.bias) {
    array b = state.const_tensor_ref(*n.bias);
    Y = add(Y, b, s);
  }

  set_output(state, n.out, std::move(Y));
}

// ----- RMSNorm -----
inline void do_rmsnorm(const Program&,
                       ExecutionState& state,
                       const RMSNormNode& n,
                       StreamOrDevice s) {
  using namespace ::mlx::core;

  const array& x = state.const_tensor_ref(n.x);
  array mu2 = mean(square(x, s), /*axis=*/-1, /*keepdims=*/true);
  array y   = multiply(x, rsqrt(mu2 + n.eps, s), s);

  array w = state.const_tensor_ref(n.weight);
  if (w.ndim() == 1) {
    auto shp = x.shape();
    for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
    w = reshape(w, shp);
  }
  set_output(state, n.out, multiply(y, w, s));
}

// ----- RoPE -----
inline void do_rope(const Program&,
                    ExecutionState& state,
                    const RopeNode& r,
                    StreamOrDevice s) {
  using namespace ::mlx::core;

  const array& Q = state.const_tensor_ref(r.q_in);
  const array& K = state.const_tensor_ref(r.k_in);
  const int offset = state.value_ref<int32_t>(r.pos);
  std::optional<array> freqs_arr = std::nullopt;
  if (r.freqs) {
    freqs_arr = state.const_tensor_ref(*r.freqs);
  }

  array Qr = fast::rope(Q, r.head_dim, r.traditional, r.base, r.scale, offset, freqs_arr, s);
  array Kr = fast::rope(K, r.head_dim, r.traditional, r.base, r.scale, offset, freqs_arr, s);

  set_output(state, r.q_out, std::move(Qr));
  set_output(state, r.k_out, std::move(Kr));
}

// ----- SDPA -----
inline void do_sdpa_fused(const Program&,
                          ExecutionState& state,
                          const SdpaNode& sdp,
                          StreamOrDevice s) {
  using namespace ::mlx::core;

  array Q = state.const_tensor_ref(sdp.q);
  array K = state.const_tensor_ref(sdp.k);
  array V = state.const_tensor_ref(sdp.v);

  std::string mask_mode = "";
  std::vector<array> mask_arrs;
  std::optional<array> sinks = std::nullopt;

  if (sdp.mask) {
    array M = state.const_tensor_ref(*sdp.mask);
    if (M.dtype() != Q.dtype()) M = astype(M, Q.dtype(), s);
    mask_arrs.push_back(std::move(M));
  }
  if (sdp.causal) mask_mode = "causal";

  array out = fast::scaled_dot_product_attention(
      Q, K, V, static_cast<float>(sdp.scale), mask_mode, mask_arrs, sinks, s
  );
  set_output(state, sdp.out, std::move(out));
}

// ----- Slice (single-axis scalar Vids {axis,start,length}) -----
inline void do_slice(const Program&,
                     ExecutionState& state,
                     const SliceNode& n,
                     StreamOrDevice s) {
  using namespace ::mlx::core;

  const array& x = state.const_tensor_ref(n.x);

  const int rank = static_cast<int>(x.ndim());
  auto resolve = [&](const std::variant<int, Vid<int>>& v) -> int {
    if (std::holds_alternative<int>(v)) return std::get<int>(v);
    return state.value_ref<int32_t>(std::get<Vid<int>>(v));
  };

  int axis  = resolve(n.axis);
  int start = resolve(n.start);
  int len   = resolve(n.length);

  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw std::out_of_range("Slice: axis out of range");
  if (len < 0) throw std::invalid_argument("Slice: length must be >= 0");

  std::vector<int> vstart(rank, 0);
  std::vector<int> vstop;
  vstop.reserve(rank);
  auto sh = x.shape();
  for (int i = 0; i < rank; ++i) vstop.push_back(static_cast<int>(sh[i]));

  const int dim = vstop[axis];
  if (start < 0) start += dim;
  start = std::max(0, std::min(start, dim));
  int stop = std::min(start + len, dim);

  vstart[axis] = start;
  vstop[axis]  = stop;

  set_output(state, n.out, slice(x, to_shape(vstart), to_shape(vstop), s));
}

// ----- SliceUpdate (single-axis scalar Vids {axis,start,length}) -----
inline void do_slice_update(const Program&,
                            ExecutionState& state,
                            const SliceUpdateNode& n,
                            StreamOrDevice s) {
  using namespace ::mlx::core;

  array& dst = state.tensor_ref(n.dst);
  const array& upd = state.const_tensor_ref(n.update);

  const int rank = static_cast<int>(dst.ndim());
  auto resolve = [&](const std::variant<int, Vid<int>>& v) -> int {
    if (std::holds_alternative<int>(v)) return std::get<int>(v);
    return state.value_ref<int32_t>(std::get<Vid<int>>(v));
  };

  int axis  = resolve(n.axis);
  int start = resolve(n.start);
  int len   = resolve(n.length);

  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) throw std::out_of_range("SliceUpdate: axis out of range");
  if (len < 0) throw std::invalid_argument("SliceUpdate: length must be >= 0");

  std::vector<int> vstart(rank, 0);
  std::vector<int> vstop;
  vstop.reserve(rank);
  auto sh = dst.shape();
  for (int i = 0; i < rank; ++i) vstop.push_back(static_cast<int>(sh[i]));

  const int dst_dim = vstop[axis];
  const int upd_dim = (axis < static_cast<int>(upd.ndim()))
                        ? static_cast<int>(upd.shape()[axis])
                        : 1;

  if (start < 0) start += dst_dim;
  start = std::max(0, std::min(start, dst_dim));
  int stop = std::min(start + len, dst_dim);

  vstart[axis] = start;
  vstop[axis]  = stop;

  dst = slice_update(dst, upd, to_shape(vstart), to_shape(vstop), s);
}

// ----- Quantized Linear -----
inline void do_quantized_linear(const Program&,
                                ExecutionState& state,
                                const QuantizedLinearNode& n,
                                StreamOrDevice s) {
  using namespace ::mlx::core;

  array X  = state.const_tensor_ref(n.x);
  array Wq = state.const_tensor_ref(n.w);
  array Sc = state.const_tensor_ref(n.scales);

  std::optional<array> Qb = std::nullopt;  // quantization biases (affine)
  if (n.biases) Qb = state.const_tensor_ref(*n.biases);

  // No implicit transpose; weights provided in correct orientation.
  array Y = quantized_matmul(
      /*x=*/X,
      /*w=*/Wq,
      /*scales=*/Sc,
      /*biases=*/Qb,
      /*transpose=*/true,
      /*group_size=*/n.group_size,
      /*bits=*/n.bits,
      /*mode=*/n.mode,
      /*stream=*/s
  );

  // Add neural bias (post-matmul) if present
  if (n.bias) {
    array b = state.const_tensor_ref(*n.bias);
    if (b.ndim() == 1) {
      auto shp = Y.shape();
      for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
      b = reshape(b, shp);
    }
    Y = add(Y, b, s);
  }

  if (n.out_dtype != DTypeId::f32)
    Y = astype(Y, to_dtype(n.out_dtype), s);

  set_output(state, n.out, std::move(Y));
}

// ----- Quantized Gather (no neural bias) -----
inline void do_quantized_gather(const Program&,
                                ExecutionState& state,
                                const QuantizedGatherNode& n,
                                StreamOrDevice s) {
  using namespace ::mlx::core;

  array ids = state.const_tensor_ref(n.ids);
  if (ids.dtype() != int32) ids = astype(ids, int32, s);

  array Wq = state.const_tensor_ref(n.table_q);
  array Sc = state.const_tensor_ref(n.scales);

  std::optional<array> Qb = std::nullopt;
  if (n.biases) Qb = state.const_tensor_ref(*n.biases);

  array Wq_sel = take(Wq, ids, /*axis=*/0, s);
  array Sc_sel = take(Sc, ids, /*axis=*/0, s);
  std::optional<array> Qb_sel = std::nullopt;
  if (Qb) Qb_sel = take(*Qb, ids, /*axis=*/0, s);

  array Y = dequantize(
      /*w=*/Wq_sel,
      /*scales=*/Sc_sel,
      /*biases=*/Qb_sel,
      /*group_size=*/n.group_size,
      /*bits=*/n.bits,
      /*mode=*/n.mode,
      /*stream=*/s
  );

  if (n.out_dtype != DTypeId::f32)
    Y = astype(Y, to_dtype(n.out_dtype), s);

  set_output(state, n.out, std::move(Y));
}

} // namespace impl

// -----------------------------------------------------------------------------
// Per-op handlers (stream-aware)
// -----------------------------------------------------------------------------
#define DECL_HANDLER(NAME, PAYLOAD) \
  static void op_##NAME(const Instr&, const Program&, ExecutionState&, StreamOrDevice);
OP_LIST(DECL_HANDLER)
#undef DECL_HANDLER

static void op_NOOP(const Instr&, const Program&, ExecutionState&, StreamOrDevice) {}

static void op_LINEAR(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_linear(prog, st, ins.get<LinearNode>(), s);
}
static void op_RMS_NORM(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_rmsnorm(prog, st, ins.get<RMSNormNode>(), s);
}
static void op_SDPA(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_sdpa_fused(prog, st, ins.get<SdpaNode>(), s);
}
static void op_ROPE_APPLY(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_rope(prog, st, ins.get<RopeNode>(), s);
}

static void op_ADD(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<AddNode>();
  set_output(st, n.out, add(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}
static void op_MUL(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<MulNode>();
  set_output(st, n.out, multiply(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}
static void op_SILU(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<SiluNode>();
  const auto& x = st.const_tensor_ref(n.x);
  set_output(st, n.out, multiply(x, sigmoid(x, s), s));
}

static void op_RESHAPE(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice) {
  using namespace ::mlx::core;
  const auto& n = ins.get<ReshapeNode>();
  set_output(st, n.out, reshape(st.const_tensor_ref(n.x), to_shape(n.shape)));
}
static void op_TRANSPOSE(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<TransposeNode>();
  set_output(st, n.out, transpose(st.const_tensor_ref(n.x), n.perm, s));
}
static void op_CONTIGUOUS(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<ContigNode>();
  set_output(st, n.out, copy(st.const_tensor_ref(n.x), s));
}

static void op_GATHER(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<GatherNode>();
  set_output(st, n.out, take(st.const_tensor_ref(n.table), st.const_tensor_ref(n.ids), /*axis=*/0, s));
}

static void op_SLICE(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_slice(prog, st, ins.get<SliceNode>(), s);
}

static void op_CONCAT(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<ConcatNode>();
  set_output(st, n.out, concatenate({ st.const_tensor_ref(n.a), st.const_tensor_ref(n.b) }, n.axis, s));
}

static void op_CAST(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<CastNode>();
  set_output(st, n.out, astype(st.const_tensor_ref(n.x), to_dtype(n.dtype), s));
}

static void op_FULL(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<FullNode>();
  set_output(st, n.out, full(to_shape(n.shape), n.v, to_dtype(n.dtype), s));
}
static void op_ZEROS(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<ZerosNode>();
  set_output(st, n.out, zeros(to_shape(n.shape), to_dtype(n.dtype), s));
}
static void op_ONES(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<OnesNode>();
  set_output(st, n.out, ones(to_shape(n.shape), to_dtype(n.dtype), s));
}

static void op_ARGMAX(const Instr& ins, const Program&, ExecutionState& st, StreamOrDevice s) {
  using namespace ::mlx::core;
  const auto& n = ins.get<ArgmaxNode>();
  array idx = argmax(st.const_tensor_ref(n.x), n.axis, s);
  if (idx.dtype() != int32) idx = astype(idx, int32, s);
  set_output(st, n.out, std::move(idx));
}

static void op_SLICE_UPDATE(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_slice_update(prog, st, ins.get<SliceUpdateNode>(), s);
}

static void op_QUANTIZED_GATHER(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_quantized_gather(prog, st, ins.get<QuantizedGatherNode>(), s);
}
static void op_QUANTIZED_LINEAR(const Instr& ins, const Program& prog, ExecutionState& st, StreamOrDevice s) {
  impl::do_quantized_linear(prog, st, ins.get<QuantizedLinearNode>(), s);
}

// -----------------------------------------------------------------------------
// Dense dispatch table (stream-aware)
// -----------------------------------------------------------------------------
using OpImplFn = void(*)(const Instr&, const Program&, ExecutionState&, StreamOrDevice);

#define ADD_PTR(NAME, PAYLOAD) &op_##NAME,
static constexpr std::array<OpImplFn, static_cast<size_t>(OpCode::SENTINEL)> kDispatch = {
  OP_LIST(ADD_PTR)
};
#undef ADD_PTR

static_assert(kDispatch.size() == static_cast<size_t>(OpCode::SENTINEL),
              "Dispatch table must match OpCode::SENTINEL");

// -----------------------------------------------------------------------------
// Interpreter (hot loop)
// -----------------------------------------------------------------------------
struct Interpreter {
  inline void run(const Program& prog,
                  ExecutionState& st,
                  StreamOrDevice stream = {}) const {
    // Ensure state is bound to program
    if (st.P != &prog) st.bind(prog);

    for (const auto& ins : prog.code) {
      switch (ins.op) {
        // Tiny / frequent ops: inline path
        case OpCode::NOOP:        op_NOOP(ins, prog, st, stream);        continue;
        case OpCode::ADD:         op_ADD(ins, prog, st, stream);         continue;
        case OpCode::MUL:         op_MUL(ins, prog, st, stream);         continue;
        case OpCode::SILU:        op_SILU(ins, prog, st, stream);        continue;
        case OpCode::RESHAPE:     op_RESHAPE(ins, prog, st, stream);     continue;
        case OpCode::TRANSPOSE:   op_TRANSPOSE(ins, prog, st, stream);   continue;
        case OpCode::CONTIGUOUS:  op_CONTIGUOUS(ins, prog, st, stream);  continue;
        case OpCode::CAST:        op_CAST(ins, prog, st, stream);        continue;

        // Heavier ops: table dispatch
        default: {
          const auto idx = static_cast<size_t>(ins.op);
          kDispatch[idx](ins, prog, st, stream);
          continue;
        }
      }
    }
  }
};

} // namespace executorch::mlx
