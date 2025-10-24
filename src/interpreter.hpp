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

namespace executorch::mlx {

// Local helper
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

// Alias for brevity
using StreamOrDevice = ::mlx::core::StreamOrDevice;

// -----------------------------------------------------------------------------
// Heavy op implementations (stream-aware)
// -----------------------------------------------------------------------------
namespace impl {

inline void do_matmul(const Program& prog,
                      MutableData& state,
                      const MatmulNode& n,
                      StreamOrDevice s) {
  auto A = read_tensor(prog, state, n.a);
  auto B = read_tensor(prog, state, n.b);

  if (n.ta) A = ::mlx::core::transpose(A, {-1, -2}, s);
  if (n.tb) B = ::mlx::core::transpose(B, {-1, -2}, s);

  auto Y = ::mlx::core::matmul(A, B, s);

  if (n.bias) {
    auto b = read_tensor(prog, state, *n.bias);
    if (b.ndim() == 1) {
      auto shp = Y.shape();
      for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
      b = ::mlx::core::reshape(b, shp);  // (reshape is view-like; no stream)
    }
    Y = ::mlx::core::add(Y, b, s);
  }
  write_tensor(state, n.out, std::move(Y));
}

inline void do_rmsnorm(const Program& prog,
                       MutableData& state,
                       const RMSNormNode& n,
                       StreamOrDevice s) {
  const auto& x = read_tensor(prog, state, n.x);
  auto mu2 = ::mlx::core::mean(::mlx::core::square(x, s), /*axis=*/-1, /*keepdims=*/true);
  auto y   = ::mlx::core::multiply(x, ::mlx::core::rsqrt(mu2 + n.eps, s), s);

  auto w = read_tensor(prog, state, n.weight);
  if (w.ndim() == 1) {
    auto shp = x.shape();
    for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
    w = ::mlx::core::reshape(w, shp);
  }
  write_tensor(state, n.out, ::mlx::core::multiply(y, w, s));
}

inline void do_rope(const Program& prog,
                    MutableData& state,
                    const RopeNode& r,
                    StreamOrDevice s) {
  const auto& Q = read_tensor(prog, state, r.q_in);
  const auto& K = read_tensor(prog, state, r.k_in);
  const int offset = state.i32_ref(r.pos);

  std::optional<::mlx::core::array> freqs = std::nullopt;
  if (r.freq.has_value()) {
    freqs = read_tensor(prog, state, *r.freq);
  }

  auto Qr = ::mlx::core::fast::rope(
      Q, r.head_dim, r.traditional, r.base, r.scale, offset, freqs, s
  );
  auto Kr = ::mlx::core::fast::rope(
      K, r.head_dim, r.traditional, r.base, r.scale, offset, freqs, s
  );

  write_tensor(state, r.q_out, std::move(Qr));
  write_tensor(state, r.k_out, std::move(Kr));
}

inline void do_sdpa_fused(const Program& prog,
                          MutableData& state,
                          const SdpaNode& sdp,
                          StreamOrDevice s) {
  // Q: [B, N_q, T_q, D],  K/V: [B, N_kv, T_kv, D]
  auto Q = read_tensor(prog, state, sdp.q);
  auto K = read_tensor(prog, state, sdp.k);
  auto V = read_tensor(prog, state, sdp.v);

  // Prepare MLX fast SDPA arguments
  std::string mask_mode = "";
  std::vector<::mlx::core::array> mask_arrs;
  std::optional<::mlx::core::array> sinks = std::nullopt;

  // Optional array mask
  if (sdp.mask.has_value()) {
    auto M = read_tensor(prog, state, *sdp.mask);
    if (M.dtype() != Q.dtype()) {
      M = ::mlx::core::astype(M, Q.dtype(), s);
    }
    mask_arrs.push_back(std::move(M));
  }

  if (sdp.causal) mask_mode = "causal";

  auto out = ::mlx::core::fast::scaled_dot_product_attention(
      /*queries=*/Q,
      /*keys=*/K,
      /*values=*/V,
      /*scale=*/static_cast<float>(sdp.scale),
      /*mask_mode=*/mask_mode,
      /*mask_arrs=*/mask_arrs,
      /*sinks=*/sinks,
      /*stream=*/s
  );
  write_tensor(state, sdp.out, std::move(out));
}

inline void do_slice(const Program& prog,
                     MutableData& state,
                     const SliceNode& n,
                     StreamOrDevice s) {
  const auto& x = read_tensor(prog, state, n.x);
  const ::mlx::core::Shape x_sh = x.shape();

  const auto& start_sh = state.shape_ref(n.start);
  const auto& stop_sh  = state.shape_ref(n.stop);

  std::vector<int> start(start_sh.begin(), start_sh.end());
  std::vector<int> stop (stop_sh.begin(),  stop_sh.end());

  for (int d = 0; d < static_cast<int>(stop.size()); ++d) {
    if (stop[d] == -1) stop[d] = static_cast<int>(x_sh[d]);
  }

  if (!n.strides) {
    write_tensor(state, n.out, ::mlx::core::slice(x, to_shape(start), to_shape(stop), s));
    return;
  }

  const auto& strides_sh = state.shape_ref(*n.strides);
  std::vector<int> strides(strides_sh.begin(), strides_sh.end());
  const bool all_ones = std::all_of(strides.begin(), strides.end(),
                                    [](int v){ return v == 1; });
  if (all_ones) {
    write_tensor(state, n.out, ::mlx::core::slice(x, to_shape(start), to_shape(stop), s));
    return;
  }

  write_tensor(state, n.out,
               ::mlx::core::slice(x, to_shape(start), to_shape(stop), to_shape(strides), s));
}

inline void do_slice_update(const Program&,
                            MutableData& state,
                            const SliceUpdateNode& n,
                            StreamOrDevice s) {
  auto& dst       = state.m_ref(n.dst);
  const auto& upd = read_tensor(Program{}, state, n.update);

  const auto& start_sh = state.shape_ref(n.start);
  const auto& stop_sh  = state.shape_ref(n.stop);

  std::vector<int> start(start_sh.begin(), start_sh.end());
  std::vector<int> stop (stop_sh.begin(),  stop_sh.end());

  const auto upd_sh = upd.shape();
  for (int d = 0; d < static_cast<int>(stop.size()); ++d) {
    if (stop[d] == -1) {
      const int upd_len = (d < static_cast<int>(upd_sh.size())) ? upd_sh[d] : 1;
      stop[d] = start[d] + upd_len;
    }
  }

  if (!n.strides) {
    dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop), s);
    return;
  }

  const auto& strides_sh = state.shape_ref(*n.strides);
  std::vector<int> strides(strides_sh.begin(), strides_sh.end());
  const bool all_ones = std::all_of(strides.begin(), strides.end(),
                                    [](int v){ return v == 1; });
  if (all_ones) {
    dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop), s);
    return;
  }

  dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop), to_shape(strides), s);
}

inline void do_qlinear4(const Program& prog,
                        MutableData& state,
                        const QLinear4Node& n,
                        StreamOrDevice s) {
  using namespace ::mlx::core;

  auto X = read_tensor(prog, state, n.x);
  auto W = read_tensor(prog, state, Tid{n.w});
  auto Sc= read_tensor(prog, state, Tid{n.scales});

  std::optional<array> B = std::nullopt;
  if (n.biases.has_value()) {
    B = read_tensor(prog, state, Tid{*n.biases});
  }

  auto Y = ::mlx::core::quantized_matmul(
      /*x=*/X,
      /*w=*/W,
      /*scales=*/Sc,
      /*biases=*/B,
      /*transpose=*/n.transpose,
      /*group_size=*/n.group_size,
      /*bits=*/4,
      /*mode=*/n.mode,
      /*stream=*/s
  );

  if (n.out_dtype != DTypeId::f32)
    Y = ::mlx::core::astype(Y, to_dtype(n.out_dtype), s);

  write_tensor(state, n.out, std::move(Y));
}

inline void do_qembed4(const Program& prog,
                       MutableData& state,
                       const QEmbed4Node& n,
                       StreamOrDevice s) {
  using namespace ::mlx::core;

  auto ids = read_tensor(prog, state, n.ids);
  if (ids.dtype() != int32) ids = astype(ids, int32, s);

  auto Wq = read_tensor(prog, state, Tid{n.table_q4});
  auto Sc = read_tensor(prog, state, Tid{n.scales});

  std::optional<array> Bi_full = std::nullopt;
  if (n.biases.has_value()) {
    Bi_full = read_tensor(prog, state, Tid{*n.biases});
  }

  auto Wq_sel = ::mlx::core::take(Wq, ids, /*axis=*/0, s);
  auto Sc_sel = ::mlx::core::take(Sc, ids, /*axis=*/0, s);

  std::optional<array> Bi_sel = std::nullopt;
  if (Bi_full.has_value()) {
    Bi_sel = ::mlx::core::take(*Bi_full, ids, /*axis=*/0, s);
  }

  auto Y = ::mlx::core::dequantize(
      /*w=*/Wq_sel,
      /*scales=*/Sc_sel,
      /*biases=*/Bi_sel,
      /*group_size=*/n.group_size,
      /*bits=*/4,
      /*mode=*/n.mode,
      /*stream=*/s
  );

  if (n.out_dtype != DTypeId::f32)
    Y = ::mlx::core::astype(Y, to_dtype(n.out_dtype), s);

  write_tensor(state, n.out, std::move(Y));
}

} // namespace impl

// -----------------------------------------------------------------------------
// Per-op handlers (stream-aware)
// -----------------------------------------------------------------------------
#define DECL_HANDLER(NAME, PAYLOAD) \
  static void op_##NAME(const Instr&, const Program&, MutableData&, StreamOrDevice);
LLM_OP_LIST(DECL_HANDLER)
#undef DECL_HANDLER

static void op_NOOP(const Instr&, const Program&, MutableData&, StreamOrDevice) {}

static void op_MATMUL(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_matmul(prog, st, ins.get<MatmulNode>(), s);
}

static void op_RMS_NORM(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_rmsnorm(prog, st, ins.get<RMSNormNode>(), s);
}

static void op_SDPA(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_sdpa_fused(prog, st, ins.get<SdpaNode>(), s);
}

static void op_ROPE_APPLY(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_rope(prog, st, ins.get<RopeNode>(), s);
}

static void op_ADD(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<AddNode>();
  write_tensor(st, n.out, ::mlx::core::add(read_tensor(prog, st, n.a),
                                           read_tensor(prog, st, n.b), s));
}

static void op_MUL(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<MulNode>();
  write_tensor(st, n.out, ::mlx::core::multiply(read_tensor(prog, st, n.a),
                                                read_tensor(prog, st, n.b), s));
}

static void op_SILU(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<SiluNode>();
  const auto& x = read_tensor(prog, st, n.x);
  write_tensor(st, n.out, ::mlx::core::multiply(x, ::mlx::core::sigmoid(x, s), s));
}

static void op_RESHAPE(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice) {
  const auto& n = ins.get<ReshapeNode>();
  write_tensor(st, n.out, ::mlx::core::reshape(read_tensor(prog, st, n.x), to_shape(n.shape)));
}

static void op_TRANSPOSE(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<TransposeNode>();
  write_tensor(st, n.out, ::mlx::core::transpose(read_tensor(prog, st, n.x), n.perm, s));
}

static void op_CONTIGUOUS(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<ContigNode>();
  write_tensor(st, n.out, ::mlx::core::copy(read_tensor(prog, st, n.x), s));
}

static void op_GATHER(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<GatherNode>();
  write_tensor(st, n.out, ::mlx::core::take(read_tensor(prog, st, n.table),
                                            read_tensor(prog, st, n.ids), /*axis=*/0, s));
}

static void op_SLICE(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_slice(prog, st, ins.get<SliceNode>(), s);
}

static void op_CONCAT(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<ConcatNode>();
  write_tensor(st, n.out, ::mlx::core::concatenate(
      { read_tensor(prog, st, n.a), read_tensor(prog, st, n.b) }, n.axis, s));
}

static void op_CAST(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<CastNode>();
  write_tensor(st, n.out, ::mlx::core::astype(read_tensor(prog, st, n.x), to_dtype(n.dtype), s));
}

static void op_FULL(const Instr& ins, const Program&, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<FullNode>();
  write_tensor(st, n.out, ::mlx::core::full(to_shape(n.shape), n.v, to_dtype(n.dtype), s));
}

static void op_ZEROS(const Instr& ins, const Program&, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<ZerosNode>();
  write_tensor(st, n.out, ::mlx::core::zeros(to_shape(n.shape), to_dtype(n.dtype), s));
}

static void op_ONES(const Instr& ins, const Program&, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<OnesNode>();
  write_tensor(st, n.out, ::mlx::core::ones(to_shape(n.shape), to_dtype(n.dtype), s));
}

static void op_ARGMAX(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  const auto& n = ins.get<ArgmaxNode>();
  auto idx = ::mlx::core::argmax(read_tensor(prog, st, n.x), n.axis, s);
  if (idx.dtype() != ::mlx::core::int32) {
    idx = ::mlx::core::astype(idx, ::mlx::core::int32, s);
  }
  write_tensor(st, n.out, std::move(idx));
}

static void op_SLICE_UPDATE(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_slice_update(prog, st, ins.get<SliceUpdateNode>(), s);
}

static void op_QEMBED4(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_qembed4(prog, st, ins.get<QEmbed4Node>(), s);
}

static void op_QLINEAR4(const Instr& ins, const Program& prog, MutableData& st, StreamOrDevice s) {
  impl::do_qlinear4(prog, st, ins.get<QLinear4Node>(), s);
}

// -----------------------------------------------------------------------------
// Dense dispatch table (stream-aware)
// -----------------------------------------------------------------------------
using OpImplFn = void(*)(const Instr&, const Program&, MutableData&, StreamOrDevice);

#define ADD_PTR(NAME, PAYLOAD) &op_##NAME,
static constexpr std::array<OpImplFn, static_cast<size_t>(OpCode::SENTINEL)> kDispatch = {
  LLM_OP_LIST(ADD_PTR)
};
#undef ADD_PTR

static_assert(kDispatch.size() == static_cast<size_t>(OpCode::SENTINEL),
              "Dispatch table must match OpCode::SENTINEL");

// -----------------------------------------------------------------------------
// Interpreter (hot loop) — stream plumbed through run(...)
// -----------------------------------------------------------------------------
struct Interpreter {
  inline void run(const Program& prog,
                  MutableData& st,
                  StreamOrDevice stream = {}) const {
    for (const auto& ins : prog.code) {
      switch (ins.op) {
        // --- Tiny / frequent ops: inline path ---
        case OpCode::NOOP:        op_NOOP(ins, prog, st, stream);        continue;
        case OpCode::ADD:         op_ADD(ins, prog, st, stream);         continue;
        case OpCode::MUL:         op_MUL(ins, prog, st, stream);         continue;
        case OpCode::SILU:        op_SILU(ins, prog, st, stream);        continue;
        case OpCode::RESHAPE:     op_RESHAPE(ins, prog, st, stream);     continue;
        case OpCode::TRANSPOSE:   op_TRANSPOSE(ins, prog, st, stream);   continue;
        case OpCode::CONTIGUOUS:  op_CONTIGUOUS(ins, prog, st, stream);  continue;
        case OpCode::CAST:        op_CAST(ins, prog, st, stream);        continue;

        // --- Heavy ops: table dispatch ---
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
