//  llm_builder.hpp (drop-in replacement with MidPool temp reuse)
#pragma once
#include <random>
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "id.hpp"
#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/fast.h>

#include <torch/script.h>
#include <torch/serialize.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace executorch::mlx {

// =====================================================
// Feature flags
// =====================================================

// Use quantized matmul nodes where available (weights prepared in loader)
inline constexpr bool kUseQuantizedMatMul = true;   // set true to enable Q4 paths
// Use quantized embedding (QEMBED4) where available
inline constexpr bool kUseQuantizedEmbed = false;    // set true to enable Q4 embedding

// Choose ONE compute dtype for the entire run (weights, activations, KV)
inline constexpr DTypeId kComputeDType = DTypeId::f32;  // {f16, bf16, f32}

// Map our DTypeId → MLX dtype symbol
inline auto to_mlx_dtype(DTypeId id) {
  using namespace ::mlx::core;
  switch (id) {
    case DTypeId::f16:   return float16;
    case DTypeId::bf16:  return bfloat16;
    case DTypeId::f32:   return float32;
    default:             return float32;
  }
}

// =====================================================
// Small utilities
// =====================================================
inline std::vector<float> host_uniform(size_t n, float lo=-0.02f, float hi=0.02f, uint32_t seed=42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

inline ::mlx::core::array make_array_f32(const std::vector<float>& host,
                                         const std::vector<int>& shape) {
  using namespace ::mlx::core;
  array a(host.begin(), Shape{static_cast<int>(host.size())}, float32);
  return reshape(a, Shape(shape.begin(), shape.end()));
}

// =====================================================
// LLaMA cfg
// =====================================================
struct LlamaCfg {
  int B{1};
  int T_max{4096};
  int H{32};        // query heads
  int H_kv{32};     // KV heads (GQA)
  int D_model{2048};
  int D_head{64};   // D_model = H * D_head
  int n_layers{24};
  int d_ff{5632};
  int vocab{32000};
  int T_seq{16};    // only used for initial shapes in state

  // RoPE & norm
  bool  rope_traditional{false};  // LLaMA uses "new" rope (false)
  float rope_theta{500000.0f};    // 5e5 (Llama-3); use 10000.0 for Llama-2
  int   rope_dims{-1};            // -1 => full per-head dim
  float rms_eps{1e-6f};           // LLaMA typically 1e-6
};

// ---------- Optional Q4 metadata held next to FP weights ----------
struct Q4Linear {
  std::optional<Cid> w_q4;      // quantized weights (uint8 nibbles)
  std::optional<Cid> scales;    // per-group scales
  std::optional<Cid> biases;    // optional bias (FP, if quantize() returns it)
  bool        transpose{false}; // usually false (layout baked already)
  int         group_size{64};
  std::string mode{"affine"};   // "affine" or "symmetric"
  DTypeId     out_dtype{kComputeDType};
  inline bool valid() const { return w_q4.has_value() && scales.has_value(); }
};

struct Q4Embedding {
  std::optional<Cid> table_q4;  // quantized table [vocab, Dm] (uint8-packed)
  std::optional<Cid> scales;    // per-group scales
  std::optional<Cid> biases;    // optional biases from quantize()
  int         group_size{64};
  std::string mode{"affine"};
  DTypeId     out_dtype{kComputeDType};
  inline bool valid() const { return table_q4.has_value() && scales.has_value(); }
};

struct LlamaLayerWeights {
  Cid w_rms_attn;   // [D_model]
  Cid w_rms_mlp;    // [D_model]
  Cid Wq;           // [D_model, H*D_head]
  Cid Wk;           // [D_model, H_kv*D_head]
  Cid Wv;           // [D_model, H_kv*D_head]
  Cid Wo;           // [H*D_head, D_model]
  Cid W_gate;       // [D_model, d_ff]
  Cid W_up;         // [D_model, d_ff]
  Cid W_down;       // [d_ff, D_model]

  // Optional quantized variants used when kUseQuantizedMatMul=true
  Q4Linear Wq_q4, Wk_q4, Wv_q4, Wo_q4, W_gate_q4, W_up_q4, W_down_q4;
};

struct LlamaWeights {
  // Embedding
  Cid tok_emb;            // [vocab, D_model]  (for Gather fallback)
  Q4Embedding tok_emb_q4; // Q4 metadata (for QEMBED4)

  // Output head
  Cid lm_head;      // (alias to tok_emb if tied; kept for completeness)
  Cid lm_head_T;    // [D_model, vocab]  (projection weight for logits)
  Q4Linear lm_head_q4; // quantized metadata for lm_head_T

  std::vector<LlamaLayerWeights> layers;
  Cid w_rms_final;  // [D_model]
};

// -------- KV cache IDs (no allocation ops in program!) --------
struct CacheIds {
  Mid   K_cache;   // [B,H_kv,T_max,Dh]  (MutableData only)
  Mid   V_cache;   // [B,H_kv,T_max,Dh]
  I32Id cursor;    // runtime write pointer
  // ShapeIds for slice/slice_update windows
  ShapeId sh_start, sh_stop, sh_strides;
  ShapeId sh_kv_read_start, sh_kv_read_stop;
};

struct LlamaCaches {
  std::vector<CacheIds> layer_cache; // per-layer KV
};

// -----------------------------------------------------
// MidPool: per-layer temp-id reuse (acquire → use → release)
// -----------------------------------------------------
struct MidPool {
  uint32_t base;        // start of the temp-id range
  uint32_t cap;         // number of ids reserved for this pool
  uint32_t next = 0;    // bump ptr
  std::vector<Mid> free;

  MidPool(uint32_t base_, uint32_t cap_) : base(base_), cap(cap_) {}

  inline Mid acquire() {
    if (!free.empty()) { Mid m = free.back(); free.pop_back(); return m; }
    if (next >= cap) throw std::runtime_error("MidPool exhausted");
    return Mid{static_cast<uint32_t>(base + next++)};
  }
  inline void release(Mid m) { free.push_back(m); }
};

// -----------------------------------------------------
// Reshape helpers with T inference (only ONE -1 allowed)
// -----------------------------------------------------
inline void add_reshape_BHTDh_to_BT_Dm(Program& prog, Tid x, int B, int H, int Dh, Mid out) {
  ReshapeNode r; r.x = x; r.out = out; r.shape = { B, -1, H*Dh };  // infer T only
  prog.code.push_back(make_RESHAPE(std::move(r)));
}

// -----------------------------------------------------
// Linear projections packed to heads (supports H_out != H)
// (Q4-aware via optional parameter)
// -----------------------------------------------------
inline Mid add_linear_qaware(Program& prog,
                             Tid xBTDm,
                             Cid W_fp, std::optional<Cid> b_fp,
                             const Q4Linear& q4,
                             Mid outBTDo) {
  if (kUseQuantizedMatMul && q4.valid()) {
    QLinear4Node qn;
    qn.x         = xBTDm;
    qn.w         = *q4.w_q4;
    qn.scales    = *q4.scales;
    if (q4.biases) qn.biases = *q4.biases; // optional FP bias
    qn.transpose = q4.transpose;
    qn.group_size= q4.group_size;
    qn.mode      = q4.mode;
    qn.out_dtype = q4.out_dtype;
    qn.out       = outBTDo;
    prog.code.push_back(make_QLINEAR4(std::move(qn)));
    if (b_fp) { // explicit bias if kept separate
      AddNode add; add.a = outBTDo; add.b = Tid{*b_fp}; add.out = outBTDo;
      prog.code.push_back(make_ADD(std::move(add)));
    }
  } else {
    MatmulNode mm; mm.a = xBTDm; mm.b = Tid{W_fp}; mm.out = outBTDo; mm.ta=false; mm.tb=false;
    if (b_fp) mm.bias = Tid{*b_fp};
    prog.code.push_back(make_MATMUL(std::move(mm)));
  }
  return outBTDo;
}

// pooled variant (allocates only short-lived temps via pool)
inline void add_project_and_pack_heads_pooled(Program& prog, MidPool& pool,
                                              Tid xBTDm, Cid W, std::optional<Cid> b,
                                              int B, int H_out, int Dh,
                                              Mid out /*[B,H_out,-1,Dh]*/,
                                              const Q4Linear& q4 = {}) {
  Mid tmp_proj  = pool.acquire();   // [B,-1,H_out*Dh]
  Mid tmp_BTHDh = pool.acquire();   // [B,-1,H_out,Dh]

  add_linear_qaware(prog, xBTDm, W, b, q4, tmp_proj);

  ReshapeNode r; r.x = Tid{tmp_proj}; r.out = tmp_BTHDh; r.shape = { B, -1, H_out, Dh };
  prog.code.push_back(make_RESHAPE(std::move(r)));

  TransposeNode tp; tp.x = Tid{tmp_BTHDh}; tp.out = out; tp.perm = {0,2,1,3};
  prog.code.push_back(make_TRANSPOSE(std::move(tp)));

  pool.release(tmp_BTHDh);
  pool.release(tmp_proj);
}

// -----------------------------------------------------
// RoPE with runtime position (uses I32Id cursor)
// -----------------------------------------------------
inline void add_rope_qk(Program& prog,
                        Tid q_in, Tid k_in,
                        int head_dim, I32Id pos_id,
                        Mid q_out, Mid k_out,
                        bool traditional, float theta, float scale=1.0f) {
  RopeNode rn;
  rn.q_in = q_in; rn.k_in = k_in;
  rn.q_out = q_out; rn.k_out = k_out;
  rn.head_dim = head_dim;
  rn.pos = pos_id;
  rn.traditional = traditional;
  rn.base = theta;
  rn.scale = scale;
  prog.code.push_back(make_ROPE_APPLY(std::move(rn)));
}

// -----------------------------------------------------
// SDPA wrapper
// -----------------------------------------------------
inline void add_sdpa(Program& prog, Tid q, Tid k, Tid v,
                     float scale,
                     std::optional<Tid> mask,
                     Mid out,
                     bool causal = false) {
  SdpaNode sd;
  sd.q = q; sd.k = k; sd.v = v;
  sd.out = out;
  sd.scale = scale;
  sd.mask = mask;
  sd.causal = causal;
  prog.code.push_back(make_SDPA(std::move(sd)));
}

// -----------------------------------------------------
// KV cache helper nodes (no allocation; just slice ops).
// -----------------------------------------------------
inline void add_kv_slice_update(Program& prog, const CacheIds& cache, Tid K_step, Tid V_step) {
  SliceUpdateNode k_upd; k_upd.dst=cache.K_cache; k_upd.update=K_step;
  k_upd.start=cache.sh_start; k_upd.stop=cache.sh_stop; k_upd.strides=cache.sh_strides;
  prog.code.push_back(make_SLICE_UPDATE(std::move(k_upd)));

  SliceUpdateNode v_upd; v_upd.dst=cache.V_cache; v_upd.update=V_step;
  v_upd.start=cache.sh_start; v_upd.stop=cache.sh_stop; v_upd.strides=cache.sh_strides;
  prog.code.push_back(make_SLICE_UPDATE(std::move(v_upd)));
}

inline void add_kv_read_window(Program& prog, const CacheIds& cache, Mid K_win, Mid V_win) {
  SliceNode k_sl; k_sl.x=cache.K_cache; k_sl.out=K_win;
  k_sl.start=cache.sh_kv_read_start; k_sl.stop=cache.sh_kv_read_stop;
  prog.code.push_back(make_SLICE(std::move(k_sl)));

  SliceNode v_sl; v_sl.x=cache.V_cache; v_sl.out=V_win;
  v_sl.start=cache.sh_kv_read_start; v_sl.stop=cache.sh_kv_read_stop;
  prog.code.push_back(make_SLICE(std::move(v_sl)));
}

// pooled version of output projection
inline void add_output_projection_pooled(Program& prog, MidPool& pool,
                                         Tid attn_BHTDh,
                                         Cid Wo, std::optional<Cid> bo,
                                         int B, int H, int Dh, int /*Dm*/,
                                         Mid out_BTDm,
                                         const Q4Linear& Wo_q4 = {}) {
  Mid tmp_BTHDh       = pool.acquire(); // [B,-1,H,Dh]
  Mid tmp_BTHDmpacked = pool.acquire(); // [B,-1,H*Dh]

  // [B,H,-1,Dh] -> [B,-1,H,Dh]
  TransposeNode tp; tp.x=attn_BHTDh; tp.out=tmp_BTHDh; tp.perm={0,2,1,3};
  prog.code.push_back(make_TRANSPOSE(std::move(tp)));

  // [B,-1,H,Dh] -> [B,-1,H*Dh]
  add_reshape_BHTDh_to_BT_Dm(prog, Tid{tmp_BTHDh}, B, H, Dh, tmp_BTHDmpacked);

  // [B,-1,H*Dh] x [H*Dh,Dm] -> [B,-1,Dm]
  add_linear_qaware(prog, Tid{tmp_BTHDmpacked}, Wo, bo, Wo_q4, out_BTDm);

  pool.release(tmp_BTHDmpacked);
  pool.release(tmp_BTHDh);
}

// -----------------------------------------------------
// Torch -> MLX weights (Float32 owned; downcast to global compute dtype)
// Also emits Q4 params via ::mlx::core::quantize when flag enabled.
// -----------------------------------------------------
inline LlamaWeights load_llama_weights_from_torch(const std::string& path,
                                                  ConstantData& C,
                                                  const LlamaCfg& cfg,
                                                  bool tie_lm_head_to_tok_emb = true) {
  using ::mlx::core::array;
  using ::mlx::core::Shape;
  using ::mlx::core::transpose;
  using ::mlx::core::contiguous;
  using ::mlx::core::reshape;
  using ::mlx::core::astype;
  using ::mlx::core::float32;
  using ::mlx::core::quantize;

  constexpr bool k_downcast_to_compute = true; // cast tensors to kComputeDType
  constexpr int  k_q_group = 64;
  const std::string k_q_mode = "affine";

  auto finalize_linear = [&](array a, bool transpose_linear) {
    if (transpose_linear && a.ndim() == 2) {
      a = contiguous(transpose(a, {1, 0}));
    } else {
      a = contiguous(a);
    }
    return a; // keep dtype; caller may cast to compute dtype
  };

  // --- Read file and pickle_load ---
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("load_llama_weights_from_torch: cannot open: " + path);
  ifs.seekg(0, std::ios::end);
  auto n = ifs.tellg();
  if (n <= 0) throw std::runtime_error("load_llama_weights_from_torch: empty file: " + path);
  ifs.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(n));
  if (!ifs.read(bytes.data(), n)) throw std::runtime_error("failed to read: " + path);

  at::IValue iv = torch::pickle_load(bytes);
  if (!iv.isGenericDict()) throw std::runtime_error("top-level object is not a GenericDict");

  c10::impl::GenericDict root = iv.toGenericDict();
  c10::impl::GenericDict dict = root;
  if (root.contains("state_dict")) dict = root.at("state_dict").toGenericDict();

  auto has_key = [&](const std::string& k)->bool { return dict.contains(k); };
  auto getT = [&](const std::string& k)->torch::Tensor {
    if (!dict.contains(k)) {
      int shown=0; std::ostringstream oss;
      oss << "Missing key: " << k << " (examples: ";
      for (const auto& kv : dict) { if (shown++>=12) break; oss << kv.key().toStringRef() << ", "; }
      oss << "...)";
      throw std::runtime_error(oss.str());
    }
    return dict.at(k).toTensor();
  };

  auto to_shape = [](at::IntArrayRef sizes)->Shape {
    std::vector<int> v(sizes.begin(), sizes.end());
    return Shape(v.begin(), v.end());
  };

  // --- Robust loader: CPU F32 -> MLX-owned F32 ---
  auto load_f32_owned = [&](const torch::Tensor& tin,
                            bool transpose_linear = false) -> array {
    torch::Tensor t = tin.to(torch::kCPU, /*non_blocking=*/false);
    if (t.scalar_type() != torch::kFloat32) t = t.to(torch::kFloat32);
    t = t.contiguous();
    Shape shp = to_shape(t.sizes());

    const float* p = t.data_ptr<float>();
    std::vector<float> host(p, p + t.numel());

    array a(host.begin(), shp, float32);
    a = finalize_linear(std::move(a), transpose_linear);
    if (k_downcast_to_compute) a = astype(a, to_mlx_dtype(kComputeDType));
    return a;
  };

  // Helper: cast-and-quantize an MLX array into Q4Linear slot
  auto make_q4_from_array = [&](const array& A_fp,
                                Q4Linear& slot) {
    if (!kUseQuantizedMatMul) return;
    std::vector<array> q = quantize(A_fp, /*group_size=*/k_q_group, /*bits=*/4, /*mode=*/k_q_mode, {});
    if (q.size() < 2) return; // defensive
    array qw = ::mlx::core::contiguous(q[0]);
    array sc = ::mlx::core::contiguous(q[1]);

    slot.w_q4      = C.add(std::move(qw));
    slot.scales    = C.add(std::move(sc));

    if (q.size() >= 3) {
      array qb = ::mlx::core::contiguous(q[2]);
      slot.biases = C.add(std::move(qb));
    }

    slot.transpose = false;            // already applied layout via finalize_linear
    slot.group_size= k_q_group;
    slot.mode      = k_q_mode;
    slot.out_dtype = kComputeDType;    // outputs match global compute dtype
  };

  // Helper: cast-and-quantize an MLX array into Q4Embedding slot
  auto make_qembed_from_array = [&](const array& A_fp,
                                    Q4Embedding& slot) {
    if (!kUseQuantizedEmbed) return;
    std::vector<array> q = quantize(A_fp, /*group_size=*/k_q_group, /*bits=*/4, /*mode=*/k_q_mode, {});
    if (q.size() < 2) return;
    array qw = ::mlx::core::contiguous(q[0]);
    array sc = ::mlx::core::contiguous(q[1]);

    slot.table_q4 = C.add(std::move(qw));
    slot.scales   = C.add(std::move(sc));

    if (q.size() >= 3) {
      array qb = ::mlx::core::contiguous(q[2]);
      slot.biases = C.add(std::move(qb)); // "biases" matches mlx::dequantize signature
    }

    slot.group_size = k_q_group;
    slot.mode       = k_q_mode;
    slot.out_dtype  = kComputeDType;
  };

  // --- Build weights (HF-style names) ---
  LlamaWeights W;

  const std::string k_embed =
      has_key("model.embed_tokens.weight") ? "model.embed_tokens.weight" :
      (has_key("tok_embeddings.weight")   ? "tok_embeddings.weight"
                                          : "model.embed_tokens.weight");

  // Embeddings (determine downstream activation dtype)
  {
    array E = load_f32_owned(getT(k_embed), /*transpose_linear=*/false); // [vocab, Dm]
    W.tok_emb = C.add(std::move(E));

    // Prepare quantized embedding if enabled
    if (kUseQuantizedEmbed) {
      make_qembed_from_array(C.c_ref(W.tok_emb), W.tok_emb_q4);
    }
  }

  // lm_head: always prepare [Dm, vocab] for the projection path; keep tok_emb for Gather/QEMBED
  if (tie_lm_head_to_tok_emb) {
    W.lm_head = W.tok_emb;

    // materialize a transposed, contiguous compute-dtype copy: [Dm, vocab]
    array LT = ::mlx::core::contiguous(::mlx::core::transpose(C.c_ref(W.tok_emb), {1, 0}));
    LT = ::mlx::core::astype(LT, to_mlx_dtype(kComputeDType));
    W.lm_head_T = C.add(std::move(LT));

    if (kUseQuantizedMatMul) {
      make_q4_from_array(C.c_ref(W.lm_head_T), W.lm_head_q4);
    }
  } else if (has_key("lm_head.weight")) {
    // most checkpoints store as [vocab, Dm]; loader with transpose_linear=true → [Dm, vocab]
    array L = load_f32_owned(getT("lm_head.weight"), /*transpose_linear=*/true);
    W.lm_head_T = C.add(std::move(L));
    if (kUseQuantizedMatMul) {
      make_q4_from_array(C.c_ref(W.lm_head_T), W.lm_head_q4);
    }
    W.lm_head = W.tok_emb; // not used directly, parity only
  } else {
    // fallback to tied if head missing
    W.lm_head   = W.tok_emb;
    array LT = ::mlx::core::contiguous(::mlx::core::transpose(C.c_ref(W.tok_emb), {1, 0}));
    LT = ::mlx::core::astype(LT, to_mlx_dtype(kComputeDType));
    W.lm_head_T = C.add(std::move(LT));
    if (kUseQuantizedMatMul) {
      make_q4_from_array(C.c_ref(W.lm_head_T), W.lm_head_q4);
    }
  }

  const std::string k_final_norm =
      has_key("model.norm.weight") ? "model.norm.weight" :
      (has_key("norm.weight")      ? "norm.weight"      : "model.norm.weight");
  {
    array nrm = load_f32_owned(getT(k_final_norm), /*transpose_linear=*/false);
    W.w_rms_final = C.add(std::move(nrm));
  }

  W.layers.resize(cfg.n_layers);
  for (int l=0; l<cfg.n_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    auto& Lw = W.layers[l];

    // norms
    {
      array wa = load_f32_owned(getT(p + "input_layernorm.weight"), false);
      array wm = load_f32_owned(getT(p + "post_attention_layernorm.weight"), false);
      Lw.w_rms_attn = C.add(std::move(wa));
      Lw.w_rms_mlp  = C.add(std::move(wm));
    }

    // attention
    {
      array Wq = load_f32_owned(getT(p + "self_attn.q_proj.weight"), true);
      array Wk = load_f32_owned(getT(p + "self_attn.k_proj.weight"), true);
      array Wv = load_f32_owned(getT(p + "self_attn.v_proj.weight"), true);
      array Wo = load_f32_owned(getT(p + "self_attn.o_proj.weight"), true);

      if (kUseQuantizedMatMul) {
        make_q4_from_array(Wq, Lw.Wq_q4);
        make_q4_from_array(Wk, Lw.Wk_q4);
        make_q4_from_array(Wv, Lw.Wv_q4);
        make_q4_from_array(Wo, Lw.Wo_q4);
      }

      Lw.Wq = C.add(std::move(Wq));
      Lw.Wk = C.add(std::move(Wk));
      Lw.Wv = C.add(std::move(Wv));
      Lw.Wo = C.add(std::move(Wo));
    }

    // mlp
    {
      array Wg = load_f32_owned(getT(p + "mlp.gate_proj.weight"), true);
      array Wu = load_f32_owned(getT(p + "mlp.up_proj.weight"),   true);
      array Wd = load_f32_owned(getT(p + "mlp.down_proj.weight"), true);

      if (kUseQuantizedMatMul) {
        make_q4_from_array(Wg, Lw.W_gate_q4);
        make_q4_from_array(Wu, Lw.W_up_q4);
        make_q4_from_array(Wd, Lw.W_down_q4);
      }

      Lw.W_gate = C.add(std::move(Wg));
      Lw.W_up   = C.add(std::move(Wu));
      Lw.W_down = C.add(std::move(Wd));
    }
  }

  return W;
}

// -----------------------------------------------------
// MutableData KV allocation / init (STATE ONLY!)
// -----------------------------------------------------
inline void init_llama_cache_state(MutableData& st, const LlamaCfg& cfg, const LlamaCaches& Cc) {
  using namespace ::mlx::core;
  const int B   = cfg.B;
  const int Hkv = cfg.H_kv;
  const int Tm  = cfg.T_max;
  const int Dh  = cfg.D_head;

  auto kv_dt = to_mlx_dtype(kComputeDType);

  for (int l = 0; l < cfg.n_layers; ++l) {
    const auto& ids = Cc.layer_cache[l];
    st.set_mutable_id(ids.K_cache, zeros(Shape{B, Hkv, Tm, Dh}, kv_dt));
    st.set_mutable_id(ids.V_cache, zeros(Shape{B, Hkv, Tm, Dh}, kv_dt));
  }

  const auto& shared = Cc.layer_cache[0];
  st.set_i32_id(shared.cursor, 0);
  st.set_shape_id(shared.sh_strides, {1, 1, 1, 1});
  st.set_shape_id(shared.sh_start,   {0, 0, 0, 0});
  st.set_shape_id(shared.sh_stop,    {B, Hkv, 0, Dh});
  st.set_shape_id(shared.sh_kv_read_start, {0, 0, 0, 0});
  st.set_shape_id(shared.sh_kv_read_stop,  {B, Hkv, 0, Dh});
}

// -----------------------------------------------------
// Embedding / small ops
// -----------------------------------------------------
inline Mid add_rmsnorm(Program& prog, Tid x, Cid w, Mid out, float eps=1e-6f) {
  RMSNormNode n; n.x = x; n.weight = Tid{w}; n.out = out; n.eps = eps;
  prog.code.push_back(make_RMS_NORM(std::move(n)));
  return out;
}

// Q4-aware embeddings (QEMBED4 fallback to GATHER)
inline Mid add_embeddings_qaware(Program& prog,
                                 Cid emb_fp,                // [vocab, Dm]
                                 const Q4Embedding& qemb,   // quant meta
                                 Mid input_ids,             // [B,T] i32
                                 Mid out_BTDm) {            // [B,T,Dm]
  if (kUseQuantizedEmbed && qemb.valid()) {
    QEmbed4Node n;
    n.table_q4  = *qemb.table_q4;
    n.scales    = *qemb.scales;
    if (qemb.biases) n.biases = *qemb.biases;
    n.group_size= qemb.group_size;
    n.mode      = qemb.mode;
    n.out_dtype = qemb.out_dtype;
    n.ids       = Tid{input_ids};
    n.out       = out_BTDm;
    prog.code.push_back(make_QEMBED4(std::move(n)));
  } else {
    GatherNode g; g.table = Tid{emb_fp}; g.ids = Tid{input_ids}; g.out = out_BTDm;
    prog.code.push_back(make_GATHER(std::move(g)));
  }
  return out_BTDm;
}

inline Mid add_embeddings(Program& prog, Cid emb, Mid input_ids, Mid out_BTDm) {
  GatherNode g; g.table = Tid{emb}; g.ids = Tid{input_ids}; g.out = out_BTDm;
  prog.code.push_back(make_GATHER(std::move(g)));
  return out_BTDm;
}

inline Mid add_linear(Program& prog, Tid xBTDm, Cid W, std::optional<Cid> b, Mid outBTDo) {
  MatmulNode mm; mm.a = xBTDm; mm.b = Tid{W}; mm.out = outBTDo; mm.ta=false; mm.tb=false;
  if (b) mm.bias = Tid{*b};
  prog.code.push_back(make_MATMUL(std::move(mm)));
  return outBTDo;
}

inline Mid add_silu(Program& prog, Tid x, Mid out) {
  SiluNode s; s.x = x; s.out = out; prog.code.push_back(make_SILU(std::move(s))); return out;
}
inline Mid add_mul(Program& prog, Tid a, Tid b, Mid out) {
  MulNode m; m.a=a; m.b=b; m.out=out; prog.code.push_back(make_MUL(std::move(m))); return out;
}
inline Mid add_add(Program& prog, Tid a, Tid b, Mid out) {
  AddNode m; m.a=a; m.b=b; m.out=out; prog.code.push_back(make_ADD(std::move(m))); return out;
}

// -----------------------------------------------------
// One transformer layer (T-agnostic). RoPE uses pos_id. (Pooled temps)
// -----------------------------------------------------
inline Mid add_llama_layer(Program& prog,
                           MidPool& pool,
                           const LlamaCfg& cfg, const LlamaLayerWeights& W,
                           const CacheIds& cache, I32Id pos_id,
                           Tid x_in_BTDm, Mid x_out_BTDm) {
  const int B=cfg.B, H=cfg.H, Hkv=cfg.H_kv, Dh=cfg.D_head;

  // Attn RMSNorm
  Mid x_norm = pool.acquire();
  add_rmsnorm(prog, x_in_BTDm, W.w_rms_attn, x_norm, cfg.rms_eps);

  // Q/K/V projections (packed heads)
  Mid Q = pool.acquire();
  Mid K = pool.acquire();
  Mid V = pool.acquire();

  add_project_and_pack_heads_pooled(prog, pool, Tid{x_norm}, W.Wq, std::nullopt, B, H,   Dh, Q, W.Wq_q4);
  add_project_and_pack_heads_pooled(prog, pool, Tid{x_norm}, W.Wk, std::nullopt, B, Hkv, Dh, K, W.Wk_q4);
  add_project_and_pack_heads_pooled(prog, pool, Tid{x_norm}, W.Wv, std::nullopt, B, Hkv, Dh, V, W.Wv_q4);

  pool.release(x_norm);

  // RoPE (Qr, Kr) replace Q/K then free Q/K
  Mid Qr = pool.acquire();
  Mid Kr = pool.acquire();
  const int rope_d = (cfg.rope_dims > 0 ? cfg.rope_dims : Dh);
  add_rope_qk(prog, Tid{Q}, Tid{K}, rope_d, pos_id, Qr, Kr,
              cfg.rope_traditional, cfg.rope_theta, /*scale=*/1.0f);
  pool.release(Q);
  pool.release(K);

  // KV cache write + read window
  add_kv_slice_update(prog, cache, Tid{Kr}, Tid{V});
  Mid Kwin = pool.acquire();
  Mid Vwin = pool.acquire();
  add_kv_read_window(prog, cache, Kwin, Vwin);
  pool.release(Kr);
  pool.release(V);

  // SDPA → [B,H,T,Dh] with GQA + causal mask
  Mid Attn_BHTDh = pool.acquire();
  add_sdpa(prog, Tid{Qr}, Tid{Kwin}, Tid{Vwin},
           1.0f / std::sqrt((float)Dh),
           /*mask=*/std::nullopt,
           /*out=*/Attn_BHTDh,
           /*causal=*/true);
  pool.release(Qr);
  pool.release(Kwin);
  pool.release(Vwin);

  // Output projection back to [B,T,Dm]
  Mid Attn_BTDm = pool.acquire();
  add_output_projection_pooled(prog, pool, Tid{Attn_BHTDh}, W.Wo, std::nullopt,
                               B, H, Dh, /*Dm=*/cfg.D_model, Attn_BTDm, W.Wo_q4);
  pool.release(Attn_BHTDh);

  // Residual add1
  Mid x_res1 = pool.acquire();
  add_add(prog, x_in_BTDm, Tid{Attn_BTDm}, x_res1);
  pool.release(Attn_BTDm);

  // MLP: RMSNorm → (SiLU(Wg*x)) ⊙ (Wu*x) → Wdown
  Mid x_mlp_norm = pool.acquire();
  add_rmsnorm(prog, Tid{x_res1}, W.w_rms_mlp, x_mlp_norm, cfg.rms_eps);

  Mid gate = pool.acquire();
  Mid up   = pool.acquire();
  add_linear_qaware(prog, Tid{x_mlp_norm}, W.W_gate, std::nullopt, W.W_gate_q4, gate);
  add_linear_qaware(prog, Tid{x_mlp_norm}, W.W_up,   std::nullopt, W.W_up_q4,   up);
  pool.release(x_mlp_norm);

  Mid gate_act = pool.acquire();
  add_silu(prog, Tid{gate}, gate_act);
  pool.release(gate);

  Mid prod = pool.acquire();
  add_mul(prog, Tid{gate_act}, Tid{up}, prod);
  pool.release(gate_act);
  pool.release(up);

  Mid mlp_out = pool.acquire();
  add_linear_qaware(prog, Tid{prod}, W.W_down, std::nullopt, W.W_down_q4, mlp_out);
  pool.release(prod);

  // Residual add2 → x_out
  add_add(prog, Tid{x_res1}, Tid{mlp_out}, x_out_BTDm);
  pool.release(x_res1);
  pool.release(mlp_out);

  return x_out_BTDm;
}

// ---- Graph containers
struct LlamaGraph {
  Mid input_ids;   // [B,T] i32   (T variable per run)
  Mid logits;      // [B,T,vocab]
  Mid X_embed;     // [B,T,Dm]
  Mid X;           // stream after layers
};

// -----------------------------------------------------
// Build ONE shared LLaMA graph (no KV allocation ops). T inferred.
// -----------------------------------------------------
inline LlamaGraph build_llama_shared(Program& prog,
                                     const LlamaCfg& cfg,
                                     const LlamaWeights& W,
                                     const LlamaCaches& Cc,
                                     Mid input_ids,
                                     Mid logits,
                                     int base_mid=50000) {
  (void)Cc; // used indirectly via layer calls
  LlamaGraph G;
  G.input_ids = input_ids;
  G.X_embed   = Mid{static_cast<uint32_t>(base_mid+9000)};
  G.X         = Mid{static_cast<uint32_t>(base_mid+9001)};
  G.logits    = logits;

  // Embedding (Q4 if available) + contiguous (drop if not needed)
  add_embeddings_qaware(prog, W.tok_emb, W.tok_emb_q4, G.input_ids, G.X_embed);
  { ContigNode c; c.x = Tid{G.X_embed}; c.out = G.X; prog.code.push_back(make_CONTIGUOUS(std::move(c))); }

  // Layers (each gets its own temp pool window)
  const uint32_t LAYER_BASE = base_mid + 100000;
  const uint32_t POOL_CAP   = 64;   // plenty for one layer
  for (int l=0; l<cfg.n_layers; ++l) {
    MidPool layer_pool(/*base=*/LAYER_BASE + l*1000, /*cap=*/POOL_CAP);
    Mid X_next{static_cast<uint32_t>(LAYER_BASE + l*1000 + 900)}; // persistent dest
    add_llama_layer(prog, layer_pool, cfg, W.layers[l], Cc.layer_cache[l], Cc.layer_cache[l].cursor,
                    Tid{G.X}, X_next);
    G.X = X_next;
  }

  // Tail: norm + logits (two temps max → tiny pool)
  MidPool tail_pool(base_mid + 200000, 8);
  Mid X_norm = tail_pool.acquire();
  add_rmsnorm(prog, Tid{G.X}, W.w_rms_final, X_norm, cfg.rms_eps);
  add_linear_qaware(prog, Tid{X_norm}, W.lm_head_T, std::nullopt, W.lm_head_q4, G.logits);
  tail_pool.release(X_norm);

  return G;
}

// -----------------------------------------------------
// Shared-cursor helpers (operate on the shared layer 0 IDs).
// -----------------------------------------------------
inline void set_prefill_cursor_and_shapes(MutableData& st, int B,int H,int Dh, int T_written,
                                          const CacheIds& c) {
  st.set_i32_id(c.cursor, 0);

  st.set_shape_id(c.sh_strides, {1,1,1,1});
  st.set_shape_id(c.sh_start,   {0,0,0,0});
  st.set_shape_id(c.sh_stop,    {B,H,T_written,Dh});

  st.set_shape_id(c.sh_kv_read_start, {0,0,0,0});
  st.set_shape_id(c.sh_kv_read_stop,  {B,H,T_written,Dh});
}

inline void advance_decode_cursor(MutableData& st, int B,int H,int Dh,
                                  const CacheIds& c, int cursor, int t_step) {
  const int new_cursor = cursor + t_step;
  st.set_i32_id(c.cursor, new_cursor);
  st.set_shape_id(c.sh_start, {0,0,cursor,0});
  st.set_shape_id(c.sh_stop,  {B,H,new_cursor,Dh});
  st.set_shape_id(c.sh_kv_read_start, {0,0,0,0});
  st.set_shape_id(c.sh_kv_read_stop,  {B,H,new_cursor,Dh});
}

inline void set_prefill_cursor_shared(MutableData& st,
                                      const LlamaCfg& cfg,
                                      const LlamaCaches& Cc,
                                      int T_written) {
  const auto& c = Cc.layer_cache[0];
  set_prefill_cursor_and_shapes(st, cfg.B, cfg.H_kv, cfg.D_head, T_written, c);
}

inline void advance_decode_cursor_shared(MutableData& st,
                                         const LlamaCfg& cfg,
                                         const LlamaCaches& Cc,
                                         int cursor, int t_step) {
  const auto& c = Cc.layer_cache[0];
  advance_decode_cursor(st, cfg.B, cfg.H_kv, cfg.D_head, c, cursor, t_step);
}

// After prefill, switch RoPE/KV to decode mode.
inline void begin_decode_after_prefill(MutableData& st,
                                       const LlamaCfg& cfg,
                                       const LlamaCaches& Cc,
                                       int T_written) {
  const auto& c = Cc.layer_cache[0];
  const int B   = cfg.B;
  const int Hkv = cfg.H_kv;
  const int Dh  = cfg.D_head;

  st.set_i32_id(c.cursor, T_written);                 // RoPE offset
  st.set_shape_id(c.sh_start, {0,0,T_written,0});     // next write starts here
  st.set_shape_id(c.sh_stop,  {B, Hkv, T_written, Dh});
  st.set_shape_id(c.sh_kv_read_start, {0,0,0,0});     // read [0..T_written)
  st.set_shape_id(c.sh_kv_read_stop,  {B, Hkv, T_written, Dh});
  st.set_shape_id(c.sh_strides, {1,1,1,1});
}

inline LlamaCaches make_llama_cache_ids(const LlamaCfg& cfg, int base_mid = 10000) {
  LlamaCaches Cc;
  Cc.layer_cache.resize(cfg.n_layers);

  // Layer 0 gets the first two MIDs for K/V; shapes & cursor are shared IDs.
  CacheIds layer0{
    /*K_cache*/          Mid{static_cast<uint32_t>(base_mid + 0)},
    /*V_cache*/          Mid{static_cast<uint32_t>(base_mid + 1)},
    /*cursor*/           I32Id{0},     // shared across all layers
    /*sh_start*/         ShapeId{0},   // shared across all layers
    /*sh_stop*/          ShapeId{1},   // shared across all layers
    /*sh_strides*/       ShapeId{2},   // shared across all layers
    /*sh_kv_read_start*/ ShapeId{3},   // shared across all layers
    /*sh_kv_read_stop*/  ShapeId{4}    // shared across all layers
  };
  Cc.layer_cache[0] = layer0;

  // Remaining layers: unique K/V mids per layer, but reuse the same shared IDs.
  for (int l = 1; l < cfg.n_layers; ++l) {
    Cc.layer_cache[l] = CacheIds{
      /*K_cache*/          Mid{static_cast<uint32_t>(base_mid + 2*l + 0)},
      /*V_cache*/          Mid{static_cast<uint32_t>(base_mid + 2*l + 1)},
      /*cursor*/           layer0.cursor,
      /*sh_start*/         layer0.sh_start,
      /*sh_stop*/          layer0.sh_stop,
      /*sh_strides*/       layer0.sh_strides,
      /*sh_kv_read_start*/ layer0.sh_kv_read_start,
      /*sh_kv_read_stop*/  layer0.sh_kv_read_stop
    };
  }

  return Cc;
}

} // namespace executorch::mlx
