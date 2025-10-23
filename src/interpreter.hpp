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
  switch (d) {
    case DTypeId::f16:     return ::mlx::core::float16;
    case DTypeId::f32:     return ::mlx::core::float32;
    case DTypeId::bf16:    return ::mlx::core::bfloat16;
    case DTypeId::i32:     return ::mlx::core::int32;
    case DTypeId::i64:     return ::mlx::core::int64;
    case DTypeId::u32:     return ::mlx::core::uint32;
    case DTypeId::u8:      return ::mlx::core::uint8;
    case DTypeId::boolean: return ::mlx::core::bool_;
  }
  return ::mlx::core::float32;
}

// -----------------------------------------------------------------------------
// Heavy op implementations
// -----------------------------------------------------------------------------
namespace impl {

inline void do_matmul(const Program& prog,
                      MutableData& state,
                      const MatmulNode& n) {
  auto A = read_tensor(prog, state, n.a);
  auto B = read_tensor(prog, state, n.b);

  if (n.ta) A = ::mlx::core::transpose(A, {-1, -2});
  if (n.tb) B = ::mlx::core::transpose(B, {-1, -2});

  auto Y = ::mlx::core::matmul(A, B);

  if (n.bias) {
    auto b = read_tensor(prog, state, *n.bias);
    if (b.ndim() == 1) {
      auto shp = Y.shape();
      for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
      b = ::mlx::core::reshape(b, shp);
    }
    Y = Y + b;
  }
  write_tensor(state, n.out, std::move(Y));
}

inline void do_rmsnorm(const Program& prog,
                       MutableData& state,
                       const RMSNormNode& n) {
  const auto& x = read_tensor(prog, state, n.x);
  auto mu2 = ::mlx::core::mean(::mlx::core::square(x), /*axis=*/-1, /*keepdims=*/true);
  auto y   = x * ::mlx::core::rsqrt(mu2 + n.eps);

  auto w = read_tensor(prog, state, n.weight);
  if (w.ndim() == 1) {
    auto shp = x.shape();
    for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
    w = ::mlx::core::reshape(w, shp);
  }
  write_tensor(state, n.out, y * w);
}

inline void do_rope(const Program& prog,
                    MutableData& state,
                    const RopeNode& r) {
  const auto& Q = read_tensor(prog, state, r.q_in);
  const auto& K = read_tensor(prog, state, r.k_in);
  const int offset = state.i32_ref(r.pos);

  std::optional<::mlx::core::array> freqs = std::nullopt;
  if (r.freq.has_value()) {
    freqs = read_tensor(prog, state, *r.freq);
  }

  auto Qr = ::mlx::core::fast::rope(
      Q, r.head_dim, r.traditional, r.base, r.scale, offset, freqs, {}
  );
  auto Kr = ::mlx::core::fast::rope(
      K, r.head_dim, r.traditional, r.base, r.scale, offset, freqs, {}
  );

  write_tensor(state, r.q_out, std::move(Qr));
  write_tensor(state, r.k_out, std::move(Kr));
}

inline void do_sdpa_fused(const Program& prog,
                          MutableData& state,
                          const SdpaNode& s) {
  // Q: [B, N_q, T_q, D],  K/V: [B, N_kv, T_kv, D]
  auto Q = read_tensor(prog, state, s.q);
  auto K = read_tensor(prog, state, s.k);
  auto V = read_tensor(prog, state, s.v);

  // Prepare MLX fast SDPA arguments
  std::string mask_mode = "";
  std::vector<::mlx::core::array> mask_arrs;
  std::optional<::mlx::core::array> sinks = std::nullopt;

  // Optional array mask
  if (s.mask.has_value()) {
    auto M = read_tensor(prog, state, *s.mask);
    if (M.dtype() != Q.dtype()) {
      M = ::mlx::core::astype(M, Q.dtype());  // promote to match Q/K/V dtype
    }
    mask_arrs.push_back(std::move(M));
  }

  // Optional causal mask mode
  if (s.causal) {
    mask_mode = "causal";
  }

  // Call the fused kernel
  auto out = ::mlx::core::fast::scaled_dot_product_attention(
      /*queries=*/Q,
      /*keys=*/K,
      /*values=*/V,
      /*scale=*/static_cast<float>(s.scale),
      /*mask_mode=*/mask_mode,
      /*mask_arrs=*/mask_arrs,
      /*sinks=*/sinks,
      /*stream=*/{}
  );

  write_tensor(state, s.out, std::move(out));
  return;
}


inline void do_slice(const Program& prog,
                     MutableData& state,
                     const SliceNode& n) {
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
    write_tensor(state, n.out, ::mlx::core::slice(x, to_shape(start), to_shape(stop)));
    return;
  }

  const auto& strides_sh = state.shape_ref(*n.strides);
  std::vector<int> strides(strides_sh.begin(), strides_sh.end());
  const bool all_ones = std::all_of(strides.begin(), strides.end(),
                                    [](int s){ return s == 1; });
  if (all_ones) {
    write_tensor(state, n.out, ::mlx::core::slice(x, to_shape(start), to_shape(stop)));
    return;
  }

  write_tensor(state, n.out,
               ::mlx::core::slice(x, to_shape(start), to_shape(stop), to_shape(strides)));
}

inline void do_slice_update(const Program&,
                            MutableData& state,
                            const SliceUpdateNode& n) {
  auto& dst       = state.m_ref(n.dst);
  const auto& upd = read_tensor(Program{}, state, n.update); // Program not needed for Mid

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
    dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop));
    return;
  }

  const auto& strides_sh = state.shape_ref(*n.strides);
  std::vector<int> strides(strides_sh.begin(), strides_sh.end());
  const bool all_ones = std::all_of(strides.begin(), strides.end(),
                                    [](int s){ return s == 1; });
  if (all_ones) {
    dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop));
    return;
  }

  dst = ::mlx::core::slice_update(dst, upd, to_shape(start), to_shape(stop), to_shape(strides));
}

} // namespace impl

// -----------------------------------------------------------------------------
// Per-op handlers (plain functions), auto-declared from the X-macro.
// -----------------------------------------------------------------------------
#define DECL_HANDLER(NAME, PAYLOAD) \
  static void op_##NAME(const Instr&, const Program&, MutableData&);
LLM_OP_LIST(DECL_HANDLER)
#undef DECL_HANDLER

// ---- Implementations ----
static void op_NOOP(const Instr&, const Program&, MutableData&) {}

static void op_MATMUL(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_matmul(prog, st, ins.get<MatmulNode>());
}

static void op_RMS_NORM(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_rmsnorm(prog, st, ins.get<RMSNormNode>());
}

static void op_SDPA(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_sdpa_fused(prog, st, ins.get<SdpaNode>());
}

static void op_ROPE_APPLY(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_rope(prog, st, ins.get<RopeNode>());
}

static void op_ADD(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<AddNode>();
  write_tensor(st, n.out, ::mlx::core::add(read_tensor(prog, st, n.a),
                                         read_tensor(prog, st, n.b)));
}

static void op_MUL(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<MulNode>();
  write_tensor(st, n.out, ::mlx::core::multiply(read_tensor(prog, st, n.a),
                                              read_tensor(prog, st, n.b)));
}

static void op_SILU(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<SiluNode>();
  const auto& x = read_tensor(prog, st, n.x);
  write_tensor(st, n.out, ::mlx::core::multiply(x, ::mlx::core::sigmoid(x)));
}

static void op_RESHAPE(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<ReshapeNode>();
  write_tensor(st, n.out, ::mlx::core::reshape(read_tensor(prog, st, n.x), to_shape(n.shape)));
}

static void op_TRANSPOSE(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<TransposeNode>();
  write_tensor(st, n.out, ::mlx::core::transpose(read_tensor(prog, st, n.x), n.perm));
}

static void op_CONTIGUOUS(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<ContigNode>();
  write_tensor(st, n.out, ::mlx::core::copy(read_tensor(prog, st, n.x)));
}

static void op_GATHER(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<GatherNode>();
  write_tensor(st, n.out, ::mlx::core::take(read_tensor(prog, st, n.table),
                                          read_tensor(prog, st, n.ids), /*axis=*/0));
}

static void op_SLICE(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_slice(prog, st, ins.get<SliceNode>());
}

static void op_CONCAT(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<ConcatNode>();
  write_tensor(st, n.out, ::mlx::core::concatenate(
      { read_tensor(prog, st, n.a), read_tensor(prog, st, n.b) }, n.axis));
}

static void op_CAST(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<CastNode>();
  write_tensor(st, n.out, ::mlx::core::astype(read_tensor(prog, st, n.x), to_dtype(n.dtype)));
}

static void op_FULL(const Instr& ins, const Program&, MutableData& st) {
  const auto& n = ins.get<FullNode>();
  write_tensor(st, n.out, ::mlx::core::full(to_shape(n.shape), n.v, to_dtype(n.dtype)));
}

static void op_ZEROS(const Instr& ins, const Program&, MutableData& st) {
  const auto& n = ins.get<ZerosNode>();
  write_tensor(st, n.out, ::mlx::core::zeros(to_shape(n.shape), to_dtype(n.dtype)));
}

static void op_ONES(const Instr& ins, const Program&, MutableData& st) {
  const auto& n = ins.get<OnesNode>();
  write_tensor(st, n.out, ::mlx::core::ones(to_shape(n.shape), to_dtype(n.dtype)));
}

static void op_ARGMAX(const Instr& ins, const Program& prog, MutableData& st) {
  const auto& n = ins.get<ArgmaxNode>();
  write_tensor(st, n.out, ::mlx::core::argmax(read_tensor(prog, st, n.x), n.axis));
}

static void op_SLICE_UPDATE(const Instr& ins, const Program& prog, MutableData& st) {
  impl::do_slice_update(prog, st, ins.get<SliceUpdateNode>());
}

// -----------------------------------------------------------------------------
// Dense dispatch table generated from the X-macro
// -----------------------------------------------------------------------------
using OpImplFn = void(*)(const Instr&, const Program&, MutableData&);

#define ADD_PTR(NAME, PAYLOAD) &op_##NAME,
static constexpr std::array<OpImplFn, static_cast<size_t>(OpCode::SENTINEL)> kDispatch = {
  LLM_OP_LIST(ADD_PTR)
};
#undef ADD_PTR

static_assert(kDispatch.size() == static_cast<size_t>(OpCode::SENTINEL),
              "Dispatch table must match OpCode::SENTINEL");

// -----------------------------------------------------------------------------
// Interpreter (hot loop)
// -----------------------------------------------------------------------------
struct Interpreter {
  inline void run(const Program& prog, MutableData& st) const {
    for (const auto& ins : prog.code) {
      switch (ins.op) {
        // --- Tiny / frequent ops: inline path ---
        case OpCode::NOOP:        op_NOOP(ins, prog, st);        continue;
        case OpCode::ADD:         op_ADD(ins, prog, st);         continue;
        case OpCode::MUL:         op_MUL(ins, prog, st);         continue;
        case OpCode::SILU:        op_SILU(ins, prog, st);        continue;
        case OpCode::RESHAPE:     op_RESHAPE(ins, prog, st);     continue;
        case OpCode::TRANSPOSE:   op_TRANSPOSE(ins, prog, st);   continue;
        case OpCode::CONTIGUOUS:  op_CONTIGUOUS(ins, prog, st);  continue;
        case OpCode::CAST:        op_CAST(ins, prog, st);        continue;

        // --- Heavy ops: go through dispatch table ---
        default: {
          const auto idx = static_cast<size_t>(ins.op);
          kDispatch[idx](ins, prog, st);
          continue;
        }
      }
    }
  }
};

} // namespace executorch::mlx
