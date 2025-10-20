// graph_build.hpp
#pragma once
#include "ops.hpp"
#include <optional>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

namespace llm {

// Lightweight type info you can gradually enrich as you learn shapes/dtypes.
struct ValueInfo {
  std::optional<DTypeId> dtype;
  std::vector<int>       shape;      // empty => unknown
  bool                   known() const { return dtype.has_value(); }
};

struct BuildOptions {
  bool force_argmax_i64 = true;   // Insert cast to i64 after argmax
  bool force_gather_i32 = true;   // Cast ids to i32 for GATHER
  bool prebias_matmul   = true;   // Reshape 1-D bias to [1,...,1,N]
  bool normalize_axes   = true;   // Canonicalize negative axes to positive
};

struct BuildResult {
  std::vector<Instr> program;      // Rewritten, validated, normalized
  std::vector<ValueInfo> values;   // Final value table (by Tid)
};

// Handy utility to grow a vector indexed by Tid.
template<class T>
static inline T& ensure_at(std::vector<T>& v, Tid t) {
  if (t >= v.size()) v.resize(t + 1);
  return v[t];
}

class GraphBuilder {
 public:
  GraphBuilder(const std::vector<Instr>& in, BuildOptions opt = {})
    : in_(in), opt_(opt) {
    // Reserve some space to avoid re-allocs
    out_.reserve(in_.size() + in_.size() / 8);
    values_.reserve(64);
  }

  BuildResult build() {
    for (const auto& ins : in_) {
      dispatch(ins);
    }
    return { std::move(out_), std::move(values_) };
  }

 private:
  const std::vector<Instr>& in_;
  BuildOptions opt_;
  std::vector<Instr> out_;
  std::vector<ValueInfo> values_;
  Tid next_tmp_ = 1u << 30; // high watermark for temps; or track max Tid first

  // Allocate a temporary Tid for builder-inserted nodes (casts, reshapes, etc.)
  Tid new_tmp() { return next_tmp_++; }

  // ---- Value info helpers ---------------------------------------------------

  ValueInfo& vi(Tid t) { return ensure_at(values_, t); }
  const ValueInfo* try_vi(Tid t) const {
    return (t < values_.size()) ? &values_[t] : nullptr;
  }

  // Dtype utilities
  Tid ensure_dtype(Tid src, DTypeId want, OpCode /*where*/) {
    auto& info = vi(src);
    if (info.dtype && *info.dtype == want) return src;
    // Insert a CAST node -> tmp
    Tid tmp = new_tmp();
    CastNode cn;
    cn.x = src;
    cn.out = tmp;
    cn.dtype = want;
    out_.push_back(make_cast(std::move(cn)));
    // Update type info for tmp
    auto& ti = vi(tmp);
    ti.dtype = want;
    ti.shape = info.shape; // shape preserved
    return tmp;
  }

  // Axis canonicalization: convert negative axis to positive given a rank.
  int norm_axis(int axis, int rank, const char* ctx) {
    int ax = axis;
    if (ax < 0) ax += rank;
    if (ax < 0 || ax >= rank) throw std::runtime_error(std::string(ctx) + ": axis out of range");
    return ax;
  }

  // Simple perm validation 0..rank-1 without duplicates
  void validate_perm(const std::vector<int>& perm, int rank) {
    if ((int)perm.size() != rank) throw std::runtime_error("transpose: perm rank mismatch");
    std::vector<int> seen(rank, 0);
    for (int p : perm) {
      if (p < 0 || p >= rank || seen[p]++) throw std::runtime_error("transpose: invalid perm");
    }
  }

  // ---- Inference-ish helpers (optional/partial) ----------------------------

  // Propagate a known dtype from an input if any is known; else leave unknown.
  static std::optional<DTypeId> pick_dtype(const std::initializer_list<std::optional<DTypeId>>& dts) {
    for (auto& d : dts) if (d.has_value()) return d;
    return std::nullopt;
  }

  // Update output info (dtype + shape if deducible). Keep this light; you can
  // expand as needed.
  void set_out(Tid out, std::optional<DTypeId> dt, const std::vector<int>& shape = {}) {
    auto& o = vi(out);
    if (dt) o.dtype = dt;
    if (!shape.empty()) o.shape = shape;
  }

  // ---- Op-specific rewrites/validation -------------------------------------

  void dispatch(const Instr& ins) {
    switch (ins.op) {
      case OpCode::ADD:       return on_add(ins);
      case OpCode::MUL:       return on_mul(ins);
      case OpCode::SILU:      return on_silu(ins);
      case OpCode::MATMUL:    return on_matmul(ins, /*with_bias=*/false);
      case OpCode::MATMUL_ADD:return on_matmul(ins, /*with_bias=*/true);
      case OpCode::RMS_NORM:  return on_rmsnorm(ins);
      case OpCode::SDPA:      return on_sdpa(ins);
      case OpCode::ROPE_APPLY:return on_rope(ins);
      case OpCode::RESHAPE:   return on_reshape(ins);
      case OpCode::TRANSPOSE: return on_transpose(ins);
      case OpCode::CONTIGUOUS:return on_contig(ins);
      case OpCode::GATHER:    return on_gather(ins);
      case OpCode::SLICE:     return on_slice(ins);
      case OpCode::CONCAT:    return on_concat(ins);
      case OpCode::CAST:      return on_cast(ins);
      case OpCode::FULL:      return on_full(ins);
      case OpCode::ZEROS:     return on_zeros(ins);
      case OpCode::ONES:      return on_ones(ins);
      case OpCode::ARGMAX:    return on_argmax(ins);
      default: {
        // Unknown -> pass-through
        out_.push_back(ins);
      } break;
    }
  }

  // Elementwise ADD/MUL: dtype unify (pick first known), insert casts if needed.
  void on_add(const Instr& ins) {
    const auto& n = ins.get<AddNode>();
    Tid a = n.a, b = n.b;
    auto dt = pick_dtype({ vi(a).dtype, vi(b).dtype });
    Instr x = ins;
    if (dt) {
      a = ensure_dtype(a, *dt, ins.op);
      b = ensure_dtype(b, *dt, ins.op);
      auto& xn = x.get<AddNode>();
      xn.a = a; xn.b = b;
      set_out(n.out, dt);
    }
    out_.push_back(x);
  }

  void on_mul(const Instr& ins) {
    const auto& n = ins.get<MulNode>();
    Tid a = n.a, b = n.b;
    auto dt = pick_dtype({ vi(a).dtype, vi(b).dtype });
    Instr x = ins;
    if (dt) {
      a = ensure_dtype(a, *dt, ins.op);
      b = ensure_dtype(b, *dt, ins.op);
      auto& xn = x.get<MulNode>();
      xn.a = a; xn.b = b;
      set_out(n.out, dt);
    }
    out_.push_back(x);
  }

  void on_silu(const Instr& ins) {
    const auto& n = ins.get<SiluNode>();
    set_out(n.out, vi(n.x).dtype);
    out_.push_back(ins);
  }

  void on_matmul(const Instr& ins, bool with_bias) {
    const auto& n = ins.get<MatmulNode>();
    Instr x = ins;
    // Propagate dtype (prefer A)
    auto dt = pick_dtype({ vi(n.a).dtype, vi(n.b).dtype });
    if (dt) set_out(n.out, dt);

    // Optional: bias cast/preshape hook
    if (with_bias && n.bias && opt_.prebias_matmul) {
      Tid b = *n.bias;
      if (dt) b = ensure_dtype(b, *dt, ins.op);
      auto& xn = x.get<MatmulNode>();
      xn.bias = b;
    }
    out_.push_back(x);
  }

  void on_rmsnorm(const Instr& ins) {
    const auto& n = ins.get<RMSNormNode>();
    // Output dtype = dtype(x)
    auto dx = vi(n.x).dtype;
    set_out(n.out, dx);
    // Optionally: pre-cast weight to x dtype
    Instr x = ins;
    if (dx) {
      Tid w = n.weight;
      w = ensure_dtype(w, *dx, ins.op);
      x.get<RMSNormNode>().weight = w;
    }
    out_.push_back(x);
  }

  void on_sdpa(const Instr& ins) {
    const auto& n = ins.get<SdpaNode>();
    Instr x = ins;
    // logits/mask -> cast mask to dq (dtype of Q)
    auto dq = vi(n.q).dtype;
    if (n.mask && dq) {
      Tid m = ensure_dtype(*n.mask, *dq, ins.op);
      x.get<SdpaNode>().mask = m;
    }
    // Out dtype = dtype(V) (common choice)
    set_out(n.out, vi(n.v).dtype);
    out_.push_back(x);
  }

  void on_rope(const Instr& ins) {
    const auto& n = ins.get<RopeNode>();
    set_out(n.q_out, vi(n.q_in).dtype);
    set_out(n.k_out, vi(n.k_in).dtype);
    out_.push_back(ins);
  }

  void on_reshape(const Instr& ins) {
    const auto& n = ins.get<ReshapeNode>();
    set_out(n.out, vi(n.x).dtype); // shape unknown unless statically computable
    out_.push_back(ins);
  }

  void on_transpose(const Instr& ins) {
    const auto& n = ins.get<TransposeNode>();
    if (!vi(n.x).shape.empty() && opt_.normalize_axes) {
      validate_perm(n.perm, (int)vi(n.x).shape.size());
    }
    set_out(n.out, vi(n.x).dtype);
    out_.push_back(ins);
  }

  void on_contig(const Instr& ins) {
    const auto& n = ins.get<ContigNode>();
    set_out(n.out, vi(n.x).dtype, vi(n.x).shape);
    out_.push_back(ins);
  }

  void on_gather(const Instr& ins) {
    const auto& n = ins.get<GatherNode>();
    Instr x = ins;
    if (opt_.force_gather_i32) {
      Tid ids = ensure_dtype(n.ids, DTypeId::i32, ins.op);
      x.get<GatherNode>().ids = ids;
    }
    // dtype(table) -> dtype(out)
    set_out(n.out, vi(n.table).dtype);
    out_.push_back(x);
  }

  void on_slice(const Instr& ins) {
    const auto& n = ins.get<SliceNode>();
    // Canonicalize axis if rank is known (optionally clamp start/end if sizes known)
    out_.push_back(ins);
    set_out(n.out, vi(n.x).dtype);
  }

  void on_concat(const Instr& ins) {
    const auto& n = ins.get<ConcatNode>();
    Instr x = ins;
    // Normalize axis using rank(a) if known
    if (!vi(n.a).shape.empty() && opt_.normalize_axes) {
      int rank = (int)vi(n.a).shape.size();
      int ax = norm_axis(n.axis, rank, "concat");
      x.get<ConcatNode>().axis = ax;
    }
    // dtype unify (prefer a)
    auto dt = pick_dtype({ vi(n.a).dtype, vi(n.b).dtype });
    if (dt) set_out(n.out, dt);
    out_.push_back(x);
  }

  void on_cast(const Instr& ins) {
    const auto& n = ins.get<CastNode>();
    set_out(n.out, n.dtype);
    out_.push_back(ins);
  }

  void on_full (const Instr& ins) { set_out(ins.get<FullNode>().out,  ins.get<FullNode>().dtype);  out_.push_back(ins); }
  void on_zeros(const Instr& ins) { set_out(ins.get<ZerosNode>().out, ins.get<ZerosNode>().dtype); out_.push_back(ins); }
  void on_ones (const Instr& ins) { set_out(ins.get<OnesNode>().out,  ins.get<OnesNode>().dtype);  out_.push_back(ins); }

  void on_argmax(const Instr& ins) {
    const auto& n = ins.get<ArgmaxNode>();
    Instr x = ins;
    // Normalize axis if rank is known
    if (!vi(n.x).shape.empty() && opt_.normalize_axes) {
      int rank = (int)vi(n.x).shape.size();
      x.get<ArgmaxNode>().axis = norm_axis(n.axis, rank, "argmax");
    }
    // Argmax returns integer type; enforce int64 if requested
    if (opt_.force_argmax_i64) {
      Tid tmpOut = new_tmp();
      // First: argmax node producing tmpOut (unknown int type)
      x.get<ArgmaxNode>().out = tmpOut;
      out_.push_back(x);
      // Then cast to i64 for the public out Tid
      CastNode cn;
      cn.x = tmpOut;
      cn.out = n.out;
      cn.dtype = DTypeId::i64;
      out_.push_back(make_cast(std::move(cn)));
      set_out(n.out, DTypeId::i64);
      return;
    } else {
      // leave whatever backend picks; default to i32 if you want a concrete choice
      set_out(n.out, DTypeId::i32);
      out_.push_back(x);
    }
  }
};

inline BuildResult build_and_validate(const std::vector<Instr>& program,
                                      const BuildOptions& opt = {}) {
  GraphBuilder gb(program, opt);
  return gb.build();
}

} // namespace llm
