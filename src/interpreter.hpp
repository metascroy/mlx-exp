// interpreter.hpp
#pragma once
#include "ops.hpp"

#include <vector>
#include <optional>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/fast.h>

namespace llm {

using Tensor = mlx::core::array;

static inline mlx::core::Shape to_shape(const std::vector<int>& v) {
  return mlx::core::Shape(v.begin(), v.end());
}

struct Interpreter {
  // Tensor table indexed by Tid
  std::vector<std::optional<Tensor>> T;

  // Accessors (builder guarantees presence)
  inline const Tensor& at(Tid t) const {
    if (!T[t].has_value()) {
        throw std::runtime_error("Missing tensor id " + std::to_string(t));
    }
    return T[t].value();

  }
  inline Tensor& at(Tid t) {
    if (!T[t].has_value()) {
        throw std::runtime_error("Missing tensor id " + std::to_string(t));
    }
    return T[t].value();
  }

  // Set/ensure helpers
  inline void set(Tid t, Tensor v) {
    if (t >= T.size()) T.resize(t + 1);
    T[t] = std::move(v);
  }
  inline bool has(Tid t) const {
    return t < T.size() && T[t].has_value();
  }

  // --------------------------------------------------------------------------
  // Run: program was validated/normalized by build phase.
  // --------------------------------------------------------------------------
  inline void run(const std::vector<Instr>& program) {
    for (const auto& ins : program) {
      switch (ins.op) {
        case OpCode::NOOP:
          // do nothing
          break;

        // Math / linear
        case OpCode::MATMUL: {
          const auto& n = ins.get<MatmulNode>();
          do_matmul(n, /*fused_bias=*/false);
        } break;
        case OpCode::MATMUL_ADD: {
          const auto& n = ins.get<MatmulNode>();
          do_matmul(n, /*fused_bias=*/true);
        } break;
        case OpCode::RMS_NORM: {
          const auto& n = ins.get<RMSNormNode>();
          do_rmsnorm(n);
        } break;

        // Attention
        case OpCode::SDPA: {
          const auto& n = ins.get<SdpaNode>();
          // do_sdpa(n);
          do_sdpa_fused(n);
        } break;
        case OpCode::ROPE_APPLY: {
          const auto& n = ins.get<RopeNode>();
          do_rope(n);
        } break;

        // Elementwise
        case OpCode::ADD: {
          const auto& n = ins.get<AddNode>();
          set(n.out, mlx::core::add(at(n.a), at(n.b)));
        } break;
        case OpCode::MUL: {
          const auto& n = ins.get<MulNode>();
          set(n.out, mlx::core::multiply(at(n.a), at(n.b)));
        } break;
        case OpCode::SILU: {
          const auto& n = ins.get<SiluNode>();
          const auto& x = at(n.x);
          set(n.out, mlx::core::multiply(x, mlx::core::sigmoid(x)));
        } break;

        // Shapes
        case OpCode::RESHAPE: {
          const auto& n = ins.get<ReshapeNode>();
          set(n.out, mlx::core::reshape(at(n.x), to_shape(n.shape)));
        } break;
        case OpCode::TRANSPOSE: {
          const auto& n = ins.get<TransposeNode>();
          set(n.out, mlx::core::transpose(at(n.x), n.perm));
        } break;
        case OpCode::CONTIGUOUS: {
          const auto& n = ins.get<ContigNode>();
          set(n.out, mlx::core::copy(at(n.x)));
        } break;

        // Indexing
        case OpCode::GATHER: {
          const auto& n = ins.get<GatherNode>();
          // Builder guarantees ids are int32 and axis==0
          set(n.out, mlx::core::take(at(n.table), at(n.ids), /*axis=*/0));
        } break;
        case OpCode::SLICE: {
          const auto& n = ins.get<SliceNode>();
          // Builder provides normalized axis and valid [start,end)
          set(n.out, slice_1d_fast(at(n.x), n.axis, n.start, n.end));
        } break;
        case OpCode::CONCAT: {
          const auto& n = ins.get<ConcatNode>();
          // Builder normalized axis and verified ranks/non-axis dims
          set(n.out, mlx::core::concatenate({at(n.a), at(n.b)}, n.axis));
        } break;

        // Dtype / constants
        case OpCode::CAST: {
          const auto& n = ins.get<CastNode>();
          set(n.out, mlx::core::astype(at(n.x), to_dtype(n.dtype)));
        } break;
        case OpCode::FULL: {
          const auto& n = ins.get<FullNode>();
          set(n.out, mlx::core::full(to_shape(n.shape), n.v, to_dtype(n.dtype)));
        } break;
        case OpCode::ZEROS: {
          const auto& n = ins.get<ZerosNode>();
          set(n.out, mlx::core::zeros(to_shape(n.shape), to_dtype(n.dtype)));
        } break;
        case OpCode::ONES: {
          const auto& n = ins.get<OnesNode>();
          set(n.out, mlx::core::ones(to_shape(n.shape), to_dtype(n.dtype)));
        } break;

        // Sampling
        case OpCode::ARGMAX: {
          const auto& n = ins.get<ArgmaxNode>();
          // Builder normalized axis and (optionally) inserted cast to i64
          set(n.out, mlx::core::argmax(at(n.x), n.axis));
        } break;
      }
    }
  }

private:
  // --------------------------------------------------------------------------
  // Helpers (builder already normalized/casted/validated)
  // --------------------------------------------------------------------------
  static inline mlx::core::Dtype to_dtype(DTypeId d) {
    switch (d) {
      case DTypeId::f16:     return mlx::core::float16;
      case DTypeId::f32:     return mlx::core::float32;
      case DTypeId::bf16:    return mlx::core::bfloat16;
      case DTypeId::i32:     return mlx::core::int32;
      case DTypeId::i64:     return mlx::core::int64;
      case DTypeId::u32:     return mlx::core::uint32;
      case DTypeId::u8:      return mlx::core::uint8;
      case DTypeId::boolean: return mlx::core::bool_;
    }
    // Unreachable if builder only emits supported dtypes.
    return mlx::core::float32;
  }

  // Fast slice: assumes axis/start/end already valid and in-bounds.
  static inline Tensor slice_1d_fast(const Tensor& x, int axis, int start, int end) {
    auto sh = x.shape();
    mlx::core::Shape s_start(sh.size(), 0), s_stop = sh;
    s_start[axis] = start;
    s_stop [axis] = end;   // [start, end)
    return mlx::core::slice(x, s_start, s_stop);
  }

  // --------------------------------------------------------------------------
  // Implementations
  // --------------------------------------------------------------------------
  inline void do_matmul(const MatmulNode& n, bool fused_bias) {
    auto A = at(n.a);
    auto B = at(n.b);

    if (n.ta) A = mlx::core::transpose(A, {-1, -2});
    if (n.tb) B = mlx::core::transpose(B, {-1, -2});

    auto Y = mlx::core::matmul(A, B);

    if (fused_bias && n.bias) {
      // Builder guarantees dtype match and broadcastability; keep a tiny fast-path
      // for 1-D bias to avoid mismatch when graph didn't pre-broadcast.
      auto b = at(*n.bias);
      if (b.ndim() == 1) {
        auto shp = Y.shape();
        for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
        b = mlx::core::reshape(b, shp);
      }
      Y = Y + b;
    }

    set(n.out, std::move(Y));
  }

  inline void do_rmsnorm(const RMSNormNode& n) {
    const auto& x = at(n.x);
    auto mu2 = mlx::core::mean(mlx::core::square(x), /*axis=*/-1, /*keepdims=*/true);
    auto inv = mlx::core::rsqrt(mu2 + n.eps);
    auto y   = x * inv;

    auto w = at(n.weight);
    // If builder pre-broadcasted weight, this is a no-op; otherwise fast reshape.
    if (w.ndim() == 1) {
      auto shp = x.shape();
      for (size_t i = 0; i + 1 < shp.size(); ++i) shp[i] = 1;
      w = mlx::core::reshape(w, shp);
    }
    set(n.out, y * w);
  }

  inline void do_rope(const RopeNode& r) {
    const auto& Qin = at(r.q_in);
    const int nd = (int)Qin.ndim();
    const int D  = (int)Qin.shape().back();
    const int T  = (int)Qin.shape()[nd - 2];

    // Slice cos/sin on time dim
    const int t0 = r.pos_offset;
    const int t1 = t0 + T;
    auto idxT = mlx::core::arange(t0, t1, 1, mlx::core::int32);
    auto C = mlx::core::take(at(r.cos_tbl), idxT, 0); // [T, D/2]
    auto S = mlx::core::take(at(r.sin_tbl), idxT, 0); // [T, D/2]

    // Broadcast to [..., T, D/2] via reshape
    std::vector<int> br(nd, 1);
    br[nd - 2] = T;
    br[nd - 1] = D / 2;
    C = mlx::core::reshape(C, to_shape(br));
    S = mlx::core::reshape(S, to_shape(br));
    auto idx_even = mlx::core::arange(0, D, 2, mlx::core::int32);
    auto idx_odd  = mlx::core::arange(1, D, 2, mlx::core::int32);

    auto apply_one = [&](const Tensor& X) -> Tensor {
      auto X_even = mlx::core::take(X, idx_even, /*axis=*/-1);
      auto X_odd  = mlx::core::take(X, idx_odd,  /*axis=*/-1);
      auto Xeven_p = mlx::core::subtract(X_even * C, X_odd * S);
      auto Xodd_p  = mlx::core::add     (X_odd  * C, X_even * S);

      // Interleave back to [..., D]
      auto shp = X_even.shape(); // [..., D/2]
      shp.push_back(1);
      auto E = mlx::core::reshape(Xeven_p, shp); // [..., D/2, 1]
      auto O = mlx::core::reshape(Xodd_p,  shp); // [..., D/2, 1]
      auto pairs = mlx::core::concatenate({E, O}, /*axis=*/-1); // [..., D/2, 2]
      shp.pop_back();
      shp.back() = (int)shp.back() * 2;
      return mlx::core::reshape(pairs, shp);
    };

    set(r.q_out, apply_one(at(r.q_in)));
    set(r.k_out, apply_one(at(r.k_in)));
  }

  inline void do_sdpa(const SdpaNode& s) {
    auto Q = at(s.q);
    auto K = at(s.k);
    auto V = at(s.v);

    if (s.scale != 1.0f) Q = Q * s.scale;
    auto logits = mlx::core::matmul(Q, mlx::core::transpose(K, {-1, -2}));
    if (s.mask) logits = logits + at(*s.mask);

    auto P   = mlx::core::softmax(logits, /*axis=*/-1);
    auto Out = mlx::core::matmul(P, V);
    set(s.out, std::move(Out));
  }



inline void do_sdpa_fused(const SdpaNode& s) {
  auto Q = at(s.q);
  auto K = at(s.k);
  auto V = at(s.v);

  // Expected layout: [B, H_q, T_q, D] for Q and [B, H_kv, T_kv, D] for K,V.
  // If yours is [B, T, H, D], permute first.

  // Scale is applied internally by the kernel (same meaning as manual path).
  const float scale = s.scale;

  // Mask handling:
  //  - "causal" for autoregressive
  //  - otherwise pass additive/boolean mask(s) via mask_arrs (broadcastable to [B,H,T_q,T_kv])
  std::string mask_mode;
  std::vector<mlx::core::array> mask_arrs;

  if (s.mask) {
    auto M = at(*s.mask);
    // keep dtypes aligned to avoid slow casts
    if (M.dtype() != Q.dtype()) M = mlx::core::astype(M, Q.dtype());
    mask_arrs.push_back(std::move(M));
  }

  // sinks is optional; pass {} if unused
  auto out = mlx::core::fast::scaled_dot_product_attention(
      Q, K, V,
      /*scale=*/scale,
      /*mask_mode=*/"", // TODO: support "causal"
      /*mask_arrs=*/mask_arrs,
      /*sinks=*/std::nullopt,
      /*stream=*/{}
  );

  set(s.out, std::move(out));
}


};

} // namespace llm
