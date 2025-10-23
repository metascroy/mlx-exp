//  llm_builder.hpp (drop-in replacement)
#pragma once
#include <random>
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <cassert>

#include "id.hpp"
#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"
#include <fstream>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace executorch::mlx {

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
  // Iterator constructor => MLX-owned memory (no alias to host)
  array a(host.begin(), Shape{static_cast<int>(host.size())}, float32);
  return reshape(a, Shape(shape.begin(), shape.end()));
}

// =====================================================
// Simple decoder config / LLaMA cfg
// =====================================================
struct DecoderCfg {
  int B{1};
  int H{2};
  int T_max{16};
  int D_model{32};
  int D_head{16};
  int T_prefill{4};
};

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
  bool  rope_traditional{false};  // LLaMA uses "new" rope
  float rope_theta{500000.0f};    // 5e5 (Llama-3); set 10000.0f for Llama-2
  int   rope_dims{-1};            // -1 => full per-head dim
  float rms_eps{1e-6f};           // LLaMA typically 1e-6
};

struct Weights {
  Cid Wq, Wk, Wv, Wo;
  std::optional<Cid> bq, bk, bv, bo;
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
};

struct LlamaWeights {
  Cid tok_emb;      // [vocab, D_model]
  Cid lm_head;      // alias to tok_emb (tied embeddings)
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

// Just markers in codegen; no ZEROS ops in Program.
enum class KvCacheMode { ReuseOnly };

// -----------------------------------------------------
// Reshape helpers with T inference (-1)
// -----------------------------------------------------
inline void add_reshape_BT_Dm_to_BHTDh(Program& prog, Tid x, int B,int H,int Dh, Mid out) {
  ReshapeNode r; r.x = x; r.out = out; r.shape = {B, -1, H, Dh};
  prog.code.push_back(make_RESHAPE(std::move(r)));
}

inline void add_reshape_BHTDh_to_BT_Dm(Program& prog, Tid x, int B,int H,int Dh, Mid out) {
  ReshapeNode r; r.x = x; r.out = out; r.shape = {B, -1, H*Dh};
  prog.code.push_back(make_RESHAPE(std::move(r)));
}

// -----------------------------------------------------
// Linear projections packed to heads (supports H_out != H)
// -----------------------------------------------------
inline void add_project_and_pack_heads(Program& prog, Tid xBTDm, Cid W, std::optional<Cid> b,
                                       int B,int H_out,int Dh,
                                       Mid tmp_proj /*[B,-1,H_out*Dh]*/,
                                       Mid tmp_BTHDh /*[B,-1,H_out,Dh]*/,
                                       Mid out /*[B,H_out,-1,Dh]*/) {
  // x:[B,-1,Dm], W:[Dm,H_out*Dh] -> tmp_proj:[B,-1,H_out*Dh]
  MatmulNode mm; mm.a=xBTDm; mm.b=Tid{W}; mm.out=tmp_proj; mm.ta=false; mm.tb=false;
  if (b) mm.bias = Tid{*b};
  prog.code.push_back(make_MATMUL(std::move(mm)));

  // Reshape to [B,-1,H_out,Dh]
  ReshapeNode r; r.x = Tid{tmp_proj}; r.out = tmp_BTHDh; r.shape = {B, -1, H_out, Dh};
  prog.code.push_back(make_RESHAPE(std::move(r)));

  // [B,-1,H_out,Dh] -> [B,H_out,-1,Dh]
  TransposeNode tp; tp.x = Tid{tmp_BTHDh}; tp.out = out; tp.perm = {0,2,1,3};
  prog.code.push_back(make_TRANSPOSE(std::move(tp)));
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

inline void add_output_projection(Program& prog, Tid attn_BHTDh,
                                  Cid Wo, std::optional<Cid> bo,
                                  int B,int H,int Dh, int Dm,
                                  Mid tmp_BTHDh, Mid tmp_BTHDmpacked, Mid out_BTDm) {
  // [B,H,-1,Dh] -> [B,-1,H,Dh]
  TransposeNode tp; tp.x=attn_BHTDh; tp.out=tmp_BTHDh; tp.perm={0,2,1,3};
  prog.code.push_back(make_TRANSPOSE(std::move(tp)));

  // [B,-1,H,Dh] -> [B,-1,H*Dh]
  add_reshape_BHTDh_to_BT_Dm(prog, Tid{tmp_BTHDh}, B,H,Dh, tmp_BTHDmpacked);

  // [B,-1,H*Dh] x [H*Dh,Dm] -> [B,-1,Dm]
  MatmulNode mm; mm.a=Tid{tmp_BTHDmpacked}; mm.b=Tid{Wo}; mm.out=out_BTDm;
  if (bo) mm.bias = Tid{*bo};
  prog.code.push_back(make_MATMUL(std::move(mm)));
}

// -----------------------------------------------------
// Runtime shape updates for prefill/decode (shared across layers)
// -----------------------------------------------------
inline void set_prefill_cursor_and_shapes(MutableData& st, int B,int H,int Dh, int T_written,
                                          const CacheIds& c) {
  // Cursor must be 0 for RoPE during prefill.
  st.set_i32_id(c.cursor, 0);

  st.set_shape_id(c.sh_strides, {1,1,1,1});
  st.set_shape_id(c.sh_start,   {0,0,0,0});
  st.set_shape_id(c.sh_stop,    {B,H,T_written,Dh});

  // Read window covers everything written so far
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

// -----------------------------------------------------
// Weights (bf16) – write into ConstantData via add()
// -----------------------------------------------------
inline Weights register_random_weights(ConstantData& C, const DecoderCfg& cfg, uint32_t seed=123) {
  using namespace ::mlx::core;
  const int Dm = cfg.D_model, H = cfg.H, Dh = cfg.D_head;
  auto Wq = astype(make_array_f32(host_uniform(static_cast<size_t>(Dm)*H*Dh, -0.02f, 0.02f, seed+1), {Dm, H*Dh}), bfloat16);
  auto Wk = astype(make_array_f32(host_uniform(static_cast<size_t>(Dm)*H*Dh, -0.02f, 0.02f, seed+2), {Dm, H*Dh}), bfloat16);
  auto Wv = astype(make_array_f32(host_uniform(static_cast<size_t>(Dm)*H*Dh, -0.02f, 0.02f, seed+3), {Dm, H*Dh}), bfloat16);
  auto Wo = astype(make_array_f32(host_uniform(static_cast<size_t>(H*Dh)*Dm, -0.02f, 0.02f, seed+4), {H*Dh, Dm}), bfloat16);
  return Weights{
    C.add(std::move(Wq)),
    C.add(std::move(Wk)),
    C.add(std::move(Wv)),
    C.add(std::move(Wo)),
    std::nullopt,std::nullopt,std::nullopt,std::nullopt
  };
}

inline LlamaWeights register_random_llama_weights(ConstantData& C, const LlamaCfg& cfg, uint32_t seed=1234) {
  using namespace ::mlx::core;
  LlamaWeights W;

  // Tied embeddings: only store tok_emb; lm_head aliases to tok_emb in struct.
  auto E = astype(make_array_f32(host_uniform(static_cast<size_t>(cfg.vocab)*cfg.D_model, -0.02f, 0.02f, seed+0),
                                 {cfg.vocab, cfg.D_model}), bfloat16);
  W.tok_emb = C.add(std::move(E));
  W.lm_head = W.tok_emb; // tie

  auto wfin = astype(make_array_f32(host_uniform(static_cast<size_t>(cfg.D_model), 0.9f, 1.1f, seed+2),
                                    {cfg.D_model}), bfloat16);
  W.w_rms_final = C.add(std::move(wfin));

  W.layers.resize(cfg.n_layers);
  for (int l=0; l<cfg.n_layers; ++l) {
    auto& L = W.layers[l];
    auto wa = astype(make_array_f32(host_uniform(static_cast<size_t>(cfg.D_model), 0.9f, 1.1f, seed+10*l+0), {cfg.D_model}), bfloat16);
    auto wm = astype(make_array_f32(host_uniform(static_cast<size_t>(cfg.D_model), 0.9f, 1.1f, seed+10*l+1), {cfg.D_model}), bfloat16);
    L.w_rms_attn = C.add(std::move(wa));
    L.w_rms_mlp  = C.add(std::move(wm));

    // Q uses H heads; K,V use H_kv heads (GQA)
    auto Wq = astype(make_array_f32(host_uniform((size_t)cfg.D_model * cfg.H    * cfg.D_head, -0.02f,0.02f, seed+10*l+2), {cfg.D_model, cfg.H    * cfg.D_head}), bfloat16);
    auto Wk = astype(make_array_f32(host_uniform((size_t)cfg.D_model * cfg.H_kv * cfg.D_head, -0.02f,0.02f, seed+10*l+3), {cfg.D_model, cfg.H_kv * cfg.D_head}), bfloat16);
    auto Wv = astype(make_array_f32(host_uniform((size_t)cfg.D_model * cfg.H_kv * cfg.D_head, -0.02f,0.02f, seed+10*l+4), {cfg.D_model, cfg.H_kv * cfg.D_head}), bfloat16);
    auto Wo = astype(make_array_f32(host_uniform((size_t)cfg.H * cfg.D_head * cfg.D_model,      -0.02f,0.02f, seed+10*l+5), {cfg.H * cfg.D_head, cfg.D_model}), bfloat16);
    L.Wq = C.add(std::move(Wq));
    L.Wk = C.add(std::move(Wk));
    L.Wv = C.add(std::move(Wv));
    L.Wo = C.add(std::move(Wo));

    auto Wg = astype(make_array_f32(host_uniform((size_t)cfg.D_model * cfg.d_ff, -0.02f,0.02f, seed+10*l+6), {cfg.D_model, cfg.d_ff}), bfloat16);
    auto Wu = astype(make_array_f32(host_uniform((size_t)cfg.D_model * cfg.d_ff, -0.02f,0.02f, seed+10*l+7), {cfg.D_model, cfg.d_ff}), bfloat16);
    auto Wd = astype(make_array_f32(host_uniform((size_t)cfg.d_ff * cfg.D_model, -0.02f,0.02f, seed+10*l+8), {cfg.d_ff, cfg.D_model}), bfloat16);
    L.W_gate = C.add(std::move(Wg));
    L.W_up   = C.add(std::move(Wu));
    L.W_down = C.add(std::move(Wd));
  }
  return W;
}

// -----------------------------------------------------
// Torch -> MLX weights (Float32 owned; optional downcast later)
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
  using ::mlx::core::float16;
  using ::mlx::core::bfloat16;

  constexpr bool k_downcast_to_f16 = true; // enable after bring-up

  auto dbg_stats = [&](const char* name, const array& A) {
    auto Af = astype(A, float32);
    auto nfin = ::mlx::core::sum(::mlx::core::astype(::mlx::core::isfinite(Af), float32)); nfin.eval();
    auto nnan = ::mlx::core::sum(::mlx::core::astype(::mlx::core::isnan(Af), float32));    nnan.eval();
    auto ninf = ::mlx::core::sum(::mlx::core::astype(::mlx::core::isinf(Af), float32));    ninf.eval();
    auto mn = ::mlx::core::min(Af); mn.eval();
    auto mx = ::mlx::core::max(Af); mx.eval();

    double total = 1.0;
    for (int d : A.shape()) total *= d;
    std::cout << "[load] " << name
              << " finites=" << nfin.item<float>() << "/" << total
              << " nan=" << nnan.item<float>() << " inf=" << ninf.item<float>()
              << " min=" << mn.item<float>() << " max=" << mx.item<float>() << "\n";
  };

  auto finalize_linear = [&](array a, bool transpose_linear) {
    if (transpose_linear && a.ndim() == 2) {
      a = contiguous(transpose(a, {1, 0}));
    } else {
      a = contiguous(a);
    }
    if (k_downcast_to_f16) a = astype(a, float16);
    return a;
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
                            bool transpose_linear = false,
                            const char* dbg_name = "tensor") -> array {
    torch::Tensor t = tin.to(torch::kCPU, /*non_blocking=*/false);
    if (t.scalar_type() != torch::kFloat32) t = t.to(torch::kFloat32);
    t = t.contiguous();
    Shape shp = to_shape(t.sizes());

    const float* p = t.data_ptr<float>();
    std::vector<float> host(p, p + t.numel());

    array a(host.begin(), shp, float32);
    a = finalize_linear(std::move(a), transpose_linear);

    dbg_stats(dbg_name, a);
    return a;
  };

  // --- Build weights (HF-style names) ---
  LlamaWeights W;

  const std::string k_embed =
      has_key("model.embed_tokens.weight") ? "model.embed_tokens.weight" :
      (has_key("tok_embeddings.weight")   ? "tok_embeddings.weight"
                                          : "model.embed_tokens.weight");

  // Embeddings
  {
    array E = load_f32_owned(getT(k_embed), /*transpose_linear=*/false, "tok_emb");
    W.tok_emb = C.add(std::move(E));
  }

  if (tie_lm_head_to_tok_emb) {
    W.lm_head = W.tok_emb;
  } else if (has_key("lm_head.weight")) {
    array L = load_f32_owned(getT("lm_head.weight"), /*transpose_linear=*/true, "lm_head");
    W.lm_head = C.add(std::move(L));
  } else {
    W.lm_head = W.tok_emb;
  }

  const std::string k_final_norm =
      has_key("model.norm.weight") ? "model.norm.weight" :
      (has_key("norm.weight")      ? "norm.weight"      : "model.norm.weight");
  {
    array nrm = load_f32_owned(getT(k_final_norm), /*transpose_linear=*/false, "final_norm");
    W.w_rms_final = C.add(std::move(nrm));
  }

  W.layers.resize(cfg.n_layers);
  for (int l=0; l<cfg.n_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    auto& Lw = W.layers[l];

    // norms
    Lw.w_rms_attn = C.add(load_f32_owned(getT(p + "input_layernorm.weight"), false,
                                         ("L"+std::to_string(l)+"_rms_attn").c_str()));
    Lw.w_rms_mlp  = C.add(load_f32_owned(getT(p + "post_attention_layernorm.weight"), false,
                                         ("L"+std::to_string(l)+"_rms_mlp").c_str()));

    // attention
    Lw.Wq = C.add(load_f32_owned(getT(p + "self_attn.q_proj.weight"), true,
                                 ("L"+std::to_string(l)+"_Wq").c_str()));
    Lw.Wk = C.add(load_f32_owned(getT(p + "self_attn.k_proj.weight"), true,
                                 ("L"+std::to_string(l)+"_Wk").c_str()));
    Lw.Wv = C.add(load_f32_owned(getT(p + "self_attn.v_proj.weight"), true,
                                 ("L"+std::to_string(l)+"_Wv").c_str()));
    Lw.Wo = C.add(load_f32_owned(getT(p + "self_attn.o_proj.weight"), true,
                                 ("L"+std::to_string(l)+"_Wo").c_str()));

    // mlp
    Lw.W_gate = C.add(load_f32_owned(getT(p + "mlp.gate_proj.weight"), true,
                                     ("L"+std::to_string(l)+"_Wgate").c_str()));
    Lw.W_up   = C.add(load_f32_owned(getT(p + "mlp.up_proj.weight"),   true,
                                     ("L"+std::to_string(l)+"_Wup").c_str()));
    Lw.W_down = C.add(load_f32_owned(getT(p + "mlp.down_proj.weight"), true,
                                     ("L"+std::to_string(l)+"_Wdown").c_str()));
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

  for (int l = 0; l < cfg.n_layers; ++l) {
    const auto& ids = Cc.layer_cache[l];
    // Float32 for bring-up stability; switch to bfloat16 once verified
    st.set_mutable_id(ids.K_cache, zeros(Shape{B, Hkv, Tm, Dh}, float32));
    st.set_mutable_id(ids.V_cache, zeros(Shape{B, Hkv, Tm, Dh}, float32));
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
// One transformer layer (T-agnostic). RoPE uses pos_id.
// -----------------------------------------------------
inline Mid add_llama_layer(Program& prog,
                           const LlamaCfg& cfg, const LlamaLayerWeights& W,
                           const CacheIds& cache, I32Id pos_id,
                           Tid x_in_BTDm, Mid x_out_BTDm,
                           int base_mid) {
  const int B=cfg.B, H=cfg.H, Hkv=cfg.H_kv, Dh=cfg.D_head, Dm=cfg.D_model;

  // Attn RMSNorm
  Mid x_norm{static_cast<uint32_t>(base_mid+0)};
  add_rmsnorm(prog, x_in_BTDm, W.w_rms_attn, x_norm, cfg.rms_eps);

  // Q/K/V → packed heads (T inferred). Q uses H, K/V use Hkv.
  Mid q_proj{static_cast<uint32_t>(base_mid+1)}, k_proj{static_cast<uint32_t>(base_mid+2)}, v_proj{static_cast<uint32_t>(base_mid+3)};
  Mid q_BTHDh{static_cast<uint32_t>(base_mid+4)}, k_BTHDh{static_cast<uint32_t>(base_mid+5)}, v_BTHDh{static_cast<uint32_t>(base_mid+6)};
  Mid Q{static_cast<uint32_t>(base_mid+7)}, K{static_cast<uint32_t>(base_mid+8)}, V{static_cast<uint32_t>(base_mid+9)};

  add_project_and_pack_heads(prog, Tid{x_norm}, W.Wq, std::nullopt, B, H,   Dh, q_proj, q_BTHDh, Q);
  add_project_and_pack_heads(prog, Tid{x_norm}, W.Wk, std::nullopt, B, Hkv, Dh, k_proj, k_BTHDh, K);
  add_project_and_pack_heads(prog, Tid{x_norm}, W.Wv, std::nullopt, B, Hkv, Dh, v_proj, v_BTHDh, V);

  // RoPE (runtime offset)
  Mid Qr{static_cast<uint32_t>(base_mid+10)}, Kr{static_cast<uint32_t>(base_mid+11)};
  const int rope_d = (cfg.rope_dims > 0 ? cfg.rope_dims : Dh);
  add_rope_qk(prog, Tid{Q}, Tid{K}, rope_d, pos_id, Qr, Kr,
              cfg.rope_traditional, cfg.rope_theta, /*scale=*/1.0f);

  // KV cache write + read window
  add_kv_slice_update(prog, cache, Tid{Kr}, Tid{V});
  Mid Kwin{static_cast<uint32_t>(base_mid+12)}, Vwin{static_cast<uint32_t>(base_mid+13)};
  add_kv_read_window(prog, cache, Kwin, Vwin);

  // SDPA → [B,H,T,Dh] with GQA (Q: H, K/V: Hkv) + causal mask
  Mid Attn_BHTDh{static_cast<uint32_t>(base_mid+14)};
  add_sdpa(prog, Tid{Qr}, Tid{Kwin}, Tid{Vwin},
           1.0f / std::sqrt((float)Dh),
           /*mask=*/std::nullopt,
           /*out=*/Attn_BHTDh,
           /*causal=*/true);

  // Output projection back to [B,T,Dm] (T inferred)
  Mid tmp_BTHDh{static_cast<uint32_t>(base_mid+15)}, tmp_BTHDm{static_cast<uint32_t>(base_mid+16)}, Attn_BTDm{static_cast<uint32_t>(base_mid+17)};
  add_output_projection(prog, Tid{Attn_BHTDh}, W.Wo, std::nullopt, B, H, Dh, Dm, tmp_BTHDh, tmp_BTHDm, Attn_BTDm);

  // Residual add1
  Mid x_res1{static_cast<uint32_t>(base_mid+18)};
  add_add(prog, x_in_BTDm, Tid{Attn_BTDm}, x_res1);

  // MLP: RMSNorm → (SiLU(Wg*x)) ⊙ (Wu*x) → Wdown
  Mid x_mlp_norm{static_cast<uint32_t>(base_mid+19)};
  add_rmsnorm(prog, Tid{x_res1}, W.w_rms_mlp, x_mlp_norm, cfg.rms_eps);

  Mid gate{static_cast<uint32_t>(base_mid+20)}, up{static_cast<uint32_t>(base_mid+21)}, gate_act{static_cast<uint32_t>(base_mid+22)}, prod{static_cast<uint32_t>(base_mid+23)};
  add_linear(prog, Tid{x_mlp_norm}, W.W_gate, std::nullopt, gate);     // [B,T,Dff]
  add_linear(prog, Tid{x_mlp_norm}, W.W_up,   std::nullopt, up);       // [B,T,Dff]
  add_silu  (prog, Tid{gate}, gate_act);
  add_mul   (prog, Tid{gate_act}, Tid{up}, prod);

  Mid mlp_out{static_cast<uint32_t>(base_mid+24)};
  add_linear(prog, Tid{prod}, W.W_down, std::nullopt, mlp_out);        // [B,T,Dm]

  // Residual add2 → x_out
  add_add(prog, Tid{x_res1}, Tid{mlp_out}, x_out_BTDm);
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
// Build ONE shared LLaMA graph (no KV allocation ops).
// T inferred from input_ids each run.
// -----------------------------------------------------
inline LlamaGraph build_llama_shared(Program& prog,
                                     const LlamaCfg& cfg,
                                     const LlamaWeights& W,
                                     const LlamaCaches& Cc,
                                     Mid input_ids,  // provided by caller
                                     Mid logits,     // provided by caller
                                     int base_mid=50000) {
  LlamaGraph G;
  G.input_ids = input_ids;
  G.X_embed   = Mid{static_cast<uint32_t>(base_mid+9000)};
  G.X         = Mid{static_cast<uint32_t>(base_mid+9001)};
  G.logits    = logits;

  // Embedding + contiguous
  add_embeddings(prog, W.tok_emb, G.input_ids, G.X_embed);
  { ContigNode c; c.x = Tid{G.X_embed}; c.out = G.X; prog.code.push_back(make_CONTIGUOUS(std::move(c))); }

  // Layers (T-agnostic)
  int layer_base = base_mid + 100000;
  for (int l=0; l<cfg.n_layers; ++l) {
    Mid X_next{static_cast<uint32_t>(layer_base + l*1000 + 1)};
    add_llama_layer(prog, cfg, W.layers[l], Cc.layer_cache[l], Cc.layer_cache[l].cursor,
                    Tid{G.X}, X_next, /*base_mid=*/layer_base + l*1000 + 100);
    G.X = X_next;
  }

  // Final RMSNorm + tied LM head: logits = X_norm @ tok_emb^T
  Mid X_norm{static_cast<uint32_t>(base_mid+9100)};
  add_rmsnorm(prog, Tid{G.X}, W.w_rms_final, X_norm, cfg.rms_eps);

  MatmulNode mm;
  mm.a  = Tid{X_norm};     // [B,T,Dm]
  mm.b  = Tid{W.tok_emb};  // [vocab,Dm]
  mm.ta = false;
  mm.tb = true;            // transpose -> [Dm,vocab]
  mm.out = G.logits;       // [B,T,vocab]
  prog.code.push_back(make_MATMUL(std::move(mm)));

  return G;
}

// -----------------------------------------------------
// Shared-cursor helpers (operate on the shared layer 0 IDs).
// -----------------------------------------------------
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
