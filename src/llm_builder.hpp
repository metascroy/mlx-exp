// llm_builder.hpp — sequential non-constant allocation + explicit ConstantData binding
#pragma once
#include <vector>
#include <string>
#include <optional>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <cmath>

#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <mlx/mlx.h>

namespace executorch::mlx {

inline constexpr bool    kUseQuantMatmul = true;
inline constexpr bool    kUseQuantEmbed  = true;
inline constexpr DTypeId kComputeDT      = DTypeId::f32;

static inline ::mlx::core::Dtype to_mlx(DTypeId d) {
  using namespace ::mlx::core;
  switch (d) {
    case DTypeId::f16:     return float16;
    case DTypeId::bf16:    return bfloat16;
    default:               return float32; // f32 and everything else fall back to f32
  }
}

// ----------------------------------------------------------------------------
// Config structs
// ----------------------------------------------------------------------------
struct LlamaCfg {
  int B{1}, T_max{4096};
  int H{32}, H_kv{32};
  int D_model{2048}, D_head{64};
  int n_layers{24}, d_ff{5632}, vocab{32000};
  bool  rope_traditional{false};
  float rope_theta{500000.f};
  int   rope_dims{-1};
  float rms_eps{1e-6f};
};

struct Q4Linear {
  std::optional<Tid> w_q4, scales, biases; // biases = quant affine term
  int         group_size{64};
  std::string mode{"affine"};
  DTypeId     out_dtype{kComputeDT};
  bool valid() const { return w_q4.has_value() && scales.has_value(); }
};

struct Q4Embedding {
  std::optional<Tid> table_q4, scales, biases;
  int         group_size{64};
  std::string mode{"affine"};
  DTypeId     out_dtype{kComputeDT};
  bool valid() const { return table_q4.has_value() && scales.has_value(); }
};

struct LlamaLayerWeights {
  Tid w_rms_attn, w_rms_mlp;
  Tid Wq, Wk, Wv, Wo;          // stored as [O, I]
  Tid W_gate, W_up, W_down;    // stored as [O, I]
  Q4Linear Wq_q4, Wk_q4, Wv_q4, Wo_q4, W_gate_q4, W_up_q4, W_down_q4;
};

struct LlamaWeights {
  Tid         tok_emb;     // [vocab, Dm]
  Q4Embedding tok_emb_q4;
  Tid      lm_head_T;      // [O, I] = [vocab, Dm]
  Q4Linear lm_head_q4;
  Tid w_rms_final;
  std::vector<LlamaLayerWeights> layers;
};

// ----------------------------------------------------------------------------
// Small op helpers
// ----------------------------------------------------------------------------
inline void add_rmsnorm(Program& P, Tid x, Tid w, Tid out, float eps) {
  RMSNormNode n; n.x=x; n.weight=w; n.out=out; n.eps=eps;
  P.code.push_back(make_RMS_NORM(std::move(n)));
}

inline void add_linear_qaware(Program& P, Tid x, Tid W, std::optional<Tid> b,
                              const Q4Linear& q, Tid out) {
  if (kUseQuantMatmul && q.valid()) {
    QuantizedLinearNode qn;
    qn.x = x;
    qn.w = *q.w_q4;            // stored [O, I]; interpreter calls transpose=true
    qn.scales = *q.scales;
    qn.biases = q.biases;      // quant affine biases (optional)
    qn.bias   = b;             // neural bias (optional)
    qn.group_size = q.group_size;
    qn.bits       = 4;
    qn.mode       = q.mode;
    qn.out_dtype  = q.out_dtype;
    qn.out        = out;
    P.code.push_back(make_QUANTIZED_LINEAR(std::move(qn)));
  } else {
    LinearNode ln;
    ln.x = x;
    ln.weight = W;             // stored [O, I]; interpreter does W^T
    ln.bias = b;
    ln.out = out;
    P.code.push_back(make_LINEAR(std::move(ln)));
  }
}

inline void add_project_pack_heads(Program& P, Tid x, Tid W, std::optional<Tid> b,
                                   int B,int Hout,int Dh, Tid outBHTDh, const Q4Linear& qmeta) {
  Tid tmp = outBHTDh;
  add_linear_qaware(P, x, W, b, qmeta, tmp);
  ReshapeNode r; r.x=tmp; r.out=tmp; r.shape={B,-1,Hout,Dh}; P.code.push_back(make_RESHAPE(std::move(r)));
  TransposeNode t; t.x=tmp; t.out=outBHTDh; t.perm={0,2,1,3}; P.code.push_back(make_TRANSPOSE(std::move(t)));
}

// NOTE: pos_scalar is a scalar runtime value => Vid<int>
inline void add_rope_qk(Program& P, Tid q_in, Tid k_in, int Dh, Vid<int> pos_scalar,
                        Tid q_out, Tid k_out, bool traditional, float theta, float scale=1.f) {
  RopeNode rn;
  rn.q_in=q_in; rn.k_in=k_in; rn.q_out=q_out; rn.k_out=k_out;
  rn.head_dim=Dh; rn.traditional=traditional; rn.base=theta; rn.scale=scale;
  rn.pos=pos_scalar;
  P.code.push_back(make_ROPE_APPLY(std::move(rn)));
}

inline void add_sdpa(Program& P, Tid q, Tid k, Tid v, float scale,
                     std::optional<Tid> mask, Tid out, bool causal) {
  SdpaNode n; n.q=q; n.k=k; n.v=v; n.out=out; n.scale=scale; n.mask=mask; n.causal=causal;
  P.code.push_back(make_SDPA(std::move(n)));
}

inline void add_kv_write_read(Program& P, const struct CacheIds& c, Tid K_step, Tid V_step, Tid K_win, Tid V_win);

// ----------------------------------------------------------------------------
// Simple allocators for non-constant slots
// ----------------------------------------------------------------------------
inline Tid alloc_tensor(Program& P) {
  Tid id{ static_cast<uint32_t>(P.num_constant_tensors + P.num_non_constant_tensors) };
  ++P.num_non_constant_tensors;
  if (P.tensor_meta.size() <= id.idx) P.tensor_meta.resize(id.idx + 1);
  return id;
}

template <typename T>
inline Vid<T> alloc_value(Program& P) {
  Vid<T> id{ static_cast<uint32_t>(P.num_non_constant_values) };
  ++P.num_non_constant_values;
  return id;
}

// ----------------------------------------------------------------------------
// One layer
// ----------------------------------------------------------------------------
inline void add_llama_layer(Program& P, const LlamaCfg& cfg, const LlamaLayerWeights& W,
                            const struct CacheIds& Cc, Vid<int> pos_scalar,
                            Tid x_in_BTDm, Tid x_out_BTDm,
                            Tid x_norm, Tid Q, Tid K, Tid V,
                            Tid Qr, Tid Kr, Tid Kwin, Tid Vwin,
                            Tid Attn_BHTDh, Tid Attn_BTDm,
                            Tid x_res1, Tid x_mlp_norm, Tid gate, Tid up, Tid gate_act, Tid prod, Tid mlp_out) {
  const int B=cfg.B, H=cfg.H, Hkv=cfg.H_kv, Dh=cfg.D_head;

  add_rmsnorm(P, x_in_BTDm, W.w_rms_attn, x_norm, cfg.rms_eps);
  add_project_pack_heads(P, x_norm, W.Wq, std::nullopt, B, H,   Dh, Q, W.Wq_q4);
  add_project_pack_heads(P, x_norm, W.Wk, std::nullopt, B, Hkv, Dh, K, W.Wk_q4);
  add_project_pack_heads(P, x_norm, W.Wv, std::nullopt, B, Hkv, Dh, V, W.Wv_q4);

  add_rope_qk(P, Q, K, (cfg.rope_dims>0?cfg.rope_dims:Dh), pos_scalar, Qr, Kr,
              cfg.rope_traditional, cfg.rope_theta, 1.0f);

  add_kv_write_read(P, Cc, Kr, V, Kwin, Vwin);

  add_sdpa(P, Qr, Kwin, Vwin, 1.0f/std::sqrt((float)Dh), std::nullopt, Attn_BHTDh, /*causal*/true);

  TransposeNode tp; tp.x=Attn_BHTDh; tp.out=Attn_BTDm; tp.perm={0,2,1,3}; P.code.push_back(make_TRANSPOSE(std::move(tp)));
  ReshapeNode r; r.x=Attn_BTDm; r.out=Attn_BTDm; r.shape={B,-1,H*Dh}; P.code.push_back(make_RESHAPE(std::move(r)));
  add_linear_qaware(P, Attn_BTDm, W.Wo, std::nullopt, W.Wo_q4, Attn_BTDm);

  { AddNode a; a.a=x_in_BTDm; a.b=Attn_BTDm; a.out=x_res1; P.code.push_back(make_ADD(std::move(a))); }

  add_rmsnorm(P, x_res1, W.w_rms_mlp, x_mlp_norm, cfg.rms_eps);
  add_linear_qaware(P, x_mlp_norm, W.W_gate, std::nullopt, W.W_gate_q4, gate);
  add_linear_qaware(P, x_mlp_norm, W.W_up,   std::nullopt, W.W_up_q4,   up);
  { SiluNode s; s.x=gate; s.out=gate_act; P.code.push_back(make_SILU(std::move(s))); }
  { MulNode m; m.a=gate_act; m.b=up; m.out=prod; P.code.push_back(make_MUL(std::move(m))); }
  add_linear_qaware(P, prod, W.W_down, std::nullopt, W.W_down_q4, mlp_out);

  { AddNode a; a.a=x_res1; a.b=mlp_out; a.out=x_out_BTDm; P.code.push_back(make_ADD(std::move(a))); }
}

// ----------------------------------------------------------------------------
// Loader (constants only) — caller owns ConstantData and passes it in
// ----------------------------------------------------------------------------
inline LlamaWeights load_llama_weights_from_torch(const std::string& path,
                                                  Program& P,
                                                  ConstantData& store,
                                                  const LlamaCfg& cfg,
                                                  bool tie_lm_head=true) {
  using namespace ::mlx::core;

  // Rebuild constant area
  store.tensors.clear();
  P.num_constant_tensors = 0;

  auto to_mlx_dtype = [](DTypeId d){ return to_mlx(d); };

  auto to_f32_owned = [&](const array& ain, bool tpose=false)->array {
    array a = ain;
    if (a.dtype() != float32) a = astype(a, float32);
    if (tpose && a.ndim()==2) a = contiguous(transpose(a, {1,0}));
    a = contiguous(a);
    if (kComputeDT != DTypeId::f32) a = astype(a, to_mlx_dtype(kComputeDT));
    return a;
  };

  auto push_const = [&](array a)->Tid {
    store.add(std::move(a));
    Tid id{ static_cast<uint32_t>(store.tensors.size() - 1) };
    P.num_constant_tensors = static_cast<uint32_t>(store.tensors.size());
    return id;
  };

  // ---------- Load .safetensors ----------
  auto tensors = load_safetensors(path).first;
  auto has = [&](const std::string& k)->bool {
    return tensors.find(k) != tensors.end();
  };
  auto get = [&](const std::string& k)->const array& {
    auto it = tensors.find(k);
    if (it == tensors.end()) throw std::runtime_error("missing key: " + k);
    return it->second;
  };

  // ---------- Read weights ----------
  LlamaWeights W;

  // Embeddings (shape: [vocab, Dm])
  const char* k_embed =
    has("model.embed_tokens.weight") ? "model.embed_tokens.weight" :
    (has("tok_embeddings.weight")    ? "tok_embeddings.weight"     :
                                      "model.embed_tokens.weight");
  W.tok_emb = push_const(to_f32_owned(get(k_embed), /*tpose=*/false));

  // Final RMSNorm
  const char* kfn =
    has("model.norm.weight") ? "model.norm.weight" :
    (has("norm.weight")      ? "norm.weight"       :
                               "model.norm.weight");
  W.w_rms_final = push_const(to_f32_owned(get(kfn), false));

  // Per-layer
  W.layers.resize(cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    auto key = [&](const std::string& s){ return "model.layers." + std::to_string(l) + "." + s; };
    auto& Lw = W.layers[l];

    // Norms
    Lw.w_rms_attn = push_const(to_f32_owned(get(key("input_layernorm.weight")), false));
    Lw.w_rms_mlp  = push_const(to_f32_owned(get(key("post_attention_layernorm.weight")), false));

    // Projections — store as [O, I]; interpreter will use transpose=true at matmul time
    Lw.Wq     = push_const(to_f32_owned(get(key("self_attn.q_proj.weight")), false));
    Lw.Wk     = push_const(to_f32_owned(get(key("self_attn.k_proj.weight")), false));
    Lw.Wv     = push_const(to_f32_owned(get(key("self_attn.v_proj.weight")), false));
    Lw.Wo     = push_const(to_f32_owned(get(key("self_attn.o_proj.weight")), false));
    Lw.W_gate = push_const(to_f32_owned(get(key("mlp.gate_proj.weight")),   false));
    Lw.W_up   = push_const(to_f32_owned(get(key("mlp.up_proj.weight")),     false));
    Lw.W_down = push_const(to_f32_owned(get(key("mlp.down_proj.weight")),   false));
  }

  // lm_head — either tie to embeddings or load explicitly
  const bool have_lm_head = has("lm_head.weight") || has("model.lm_head.weight");
  if (!tie_lm_head && have_lm_head) {
    const char* k_lm = has("lm_head.weight") ? "lm_head.weight" : "model.lm_head.weight";
    // Keep as [O, I]; interpreter transposes at runtime.
    W.lm_head_T = push_const(to_f32_owned(get(k_lm), /*tpose=*/false));
  } else {
    // Tie to embeddings (reuse [vocab, Dm])
    array LT = store.const_tensor_ref(W.tok_emb);
    if (kComputeDT != DTypeId::f32) LT = astype(LT, to_mlx_dtype(kComputeDT));
    W.lm_head_T = push_const(contiguous(LT));
  }

  // ---------- Optional quantization paths ----------
  const bool will_share_embed_q = (tie_lm_head && kUseQuantEmbed);

  if (kUseQuantMatmul) {
    auto qlin = [&](Tid src, Q4Linear& out){
      array Woi = store.const_tensor_ref(src);           // [O, I]
      auto q = quantize(Woi, /*group_size=*/64, /*bits=*/4, /*mode=*/"affine", {});
      out.w_q4    = push_const(contiguous(q[0]));
      out.scales  = push_const(contiguous(q[1]));
      if (q.size() >= 3) out.biases = push_const(contiguous(q[2]));
      out.group_size = 64; out.mode = "affine"; out.out_dtype = kComputeDT;
    };

    for (auto& Lw : W.layers) {
      qlin(Lw.Wq,     Lw.Wq_q4);
      qlin(Lw.Wk,     Lw.Wk_q4);
      qlin(Lw.Wv,     Lw.Wv_q4);
      qlin(Lw.Wo,     Lw.Wo_q4);
      qlin(Lw.W_gate, Lw.W_gate_q4);
      qlin(Lw.W_up,   Lw.W_up_q4);
      qlin(Lw.W_down, Lw.W_down_q4);
    }

    if (!will_share_embed_q) {
      qlin(W.lm_head_T, W.lm_head_q4);
    }
  }

  if (kUseQuantEmbed) {
    auto q = quantize(store.const_tensor_ref(W.tok_emb),
                      /*group_size=*/64, /*bits=*/4, /*mode=*/"affine", {});
    W.tok_emb_q4.table_q4 = push_const(contiguous(q[0]));
    W.tok_emb_q4.scales   = push_const(contiguous(q[1]));
    if (q.size() >= 3) W.tok_emb_q4.biases = push_const(contiguous(q[2]));
    W.tok_emb_q4.group_size = 64; W.tok_emb_q4.mode = "affine"; W.tok_emb_q4.out_dtype = kComputeDT;

    if (tie_lm_head && kUseQuantMatmul) {
      W.lm_head_q4.w_q4      = W.tok_emb_q4.table_q4;
      W.lm_head_q4.scales    = W.tok_emb_q4.scales;
      W.lm_head_q4.biases    = W.tok_emb_q4.biases;       // may be nullopt
      W.lm_head_q4.group_size= W.tok_emb_q4.group_size;
      W.lm_head_q4.mode      = W.tok_emb_q4.mode;
      W.lm_head_q4.out_dtype = W.tok_emb_q4.out_dtype;
    }
  }

  // Bind constants into Program
  P.bind_constants(store);
  return W;
}

// ----------------------------------------------------------------------------
// Cache / graph ids
// ----------------------------------------------------------------------------
struct CacheIds {
  Tid K_cache, V_cache;
  Vid<int> cursor;        // Scalar (position)
  // Scalar Vids for slice write window
  Vid<int> sh_axis, sh_start, sh_len;
  // Scalar Vids for slice read window
  Vid<int> rd_axis, rd_start, rd_len;
};
struct LlamaCaches { std::vector<CacheIds> by_layer; };

inline LlamaCaches make_caches(Program& P, const LlamaCfg& cfg) {
  LlamaCaches C; C.by_layer.resize(cfg.n_layers);

  // shared helpers (scalars)
  Vid<int> cursor   = alloc_value<int>(P);

  Vid<int> sh_axis  = alloc_value<int>(P);
  Vid<int> sh_start = alloc_value<int>(P);
  Vid<int> sh_len   = alloc_value<int>(P);

  Vid<int> rd_axis  = alloc_value<int>(P);
  Vid<int> rd_start = alloc_value<int>(P);
  Vid<int> rd_len   = alloc_value<int>(P);

  for (int l=0; l<cfg.n_layers; ++l) {
    Tid Kc = alloc_tensor(P);
    Tid Vc = alloc_tensor(P);
    C.by_layer[l] = { Kc, Vc, cursor, sh_axis, sh_start, sh_len, rd_axis, rd_start, rd_len };
  }
  return C;
}

struct LlamaGraphIds {
  Tid input_ids;  // input tensor slot
  Tid logits;     // output tensor slot
  Tid X_embed, X_stream;
  struct LayerScratch {
    Tid x_norm, Q, K, V, Qr, Kr, Kwin, Vwin, Attn_BHTDh, Attn_BTDm,
        x_res1, x_mlp_norm, gate, up, gate_act, prod, mlp_out, x_next;
  };
  std::vector<LayerScratch> layer;
};

inline LlamaGraphIds make_graph_ids(Program& P, const LlamaCfg& cfg) {
  LlamaGraphIds G;

  // Explicit input/output tensors (register in Program IO maps)
  G.input_ids = alloc_tensor(P);  P.add_input(G.input_ids);
  G.logits    = alloc_tensor(P);  P.add_output(G.logits);

  G.X_embed  = alloc_tensor(P);
  G.X_stream = alloc_tensor(P);

  G.layer.resize(cfg.n_layers);
  for (int l=0; l<cfg.n_layers; ++l) {
    auto& S = G.layer[l];
    S.x_norm      = alloc_tensor(P);
    S.Q           = alloc_tensor(P);
    S.K           = alloc_tensor(P);
    S.V           = alloc_tensor(P);
    S.Qr          = alloc_tensor(P);
    S.Kr          = alloc_tensor(P);
    S.Kwin        = alloc_tensor(P);
    S.Vwin        = alloc_tensor(P);
    S.Attn_BHTDh  = alloc_tensor(P);
    S.Attn_BTDm   = alloc_tensor(P);
    S.x_res1      = alloc_tensor(P);
    S.x_mlp_norm  = alloc_tensor(P);
    S.gate        = alloc_tensor(P);
    S.up          = alloc_tensor(P);
    S.gate_act    = alloc_tensor(P);
    S.prod        = alloc_tensor(P);
    S.mlp_out     = alloc_tensor(P);
    S.x_next      = alloc_tensor(P);
  }
  return G;
}

// ----------------------------------------------------------------------------
// Build graph
// ----------------------------------------------------------------------------
inline void add_kv_write_read(Program& P, const CacheIds& c, Tid K_step, Tid V_step, Tid K_win, Tid V_win) {
  (void)P;
  // Write K/V at [axis, start : start+len]
  SliceUpdateNode ku; ku.dst=c.K_cache; ku.update=K_step; ku.axis=c.sh_axis; ku.start=c.sh_start; ku.length=c.sh_len;
  SliceUpdateNode vu; vu.dst=c.V_cache; vu.update=V_step; vu.axis=c.sh_axis; vu.start=c.sh_start; vu.length=c.sh_len;
  P.code.push_back(make_SLICE_UPDATE(std::move(ku)));
  P.code.push_back(make_SLICE_UPDATE(std::move(vu)));

  // Read window into K_win/V_win
  SliceNode ks; ks.x=c.K_cache; ks.out=K_win; ks.axis=c.rd_axis; ks.start=c.rd_start; ks.length=c.rd_len;
  SliceNode vs; vs.x=c.V_cache; vs.out=V_win; vs.axis=c.rd_axis; vs.start=c.rd_start; vs.length=c.rd_len;
  P.code.push_back(make_SLICE(std::move(ks)));
  P.code.push_back(make_SLICE(std::move(vs)));
}

inline void build_llama_shared(Program& P,
                               const LlamaCfg& cfg,
                               const LlamaWeights& W,
                               const LlamaCaches& Cc,
                               const LlamaGraphIds& G) {
  if (kUseQuantEmbed && W.tok_emb_q4.valid()) {
    QuantizedGatherNode n;
    n.table_q=*W.tok_emb_q4.table_q4; n.scales=*W.tok_emb_q4.scales; n.biases=W.tok_emb_q4.biases;
    n.group_size=W.tok_emb_q4.group_size; n.bits=4; n.mode=W.tok_emb_q4.mode; n.out_dtype=W.tok_emb_q4.out_dtype;
    n.ids=G.input_ids; n.out=G.X_embed; P.code.push_back(make_QUANTIZED_GATHER(std::move(n)));
  } else {
    GatherNode g; g.table=W.tok_emb; g.ids=G.input_ids; g.out=G.X_embed; P.code.push_back(make_GATHER(std::move(g)));
  }
  { ContigNode c; c.x=G.X_embed; c.out=G.X_stream; P.code.push_back(make_CONTIGUOUS(std::move(c))); }

  for (int l=0;l<cfg.n_layers;++l) {
    const auto& C = Cc.by_layer[l];
    const Vid<int> pos_scalar = C.cursor;
    add_llama_layer(P, cfg, W.layers[l], C, pos_scalar,
                    (l==0)?G.X_stream:G.layer[l-1].x_next, G.layer[l].x_next,
                    G.layer[l].x_norm, G.layer[l].Q, G.layer[l].K, G.layer[l].V,
                    G.layer[l].Qr, G.layer[l].Kr, G.layer[l].Kwin, G.layer[l].Vwin,
                    G.layer[l].Attn_BHTDh, G.layer[l].Attn_BTDm,
                    G.layer[l].x_res1, G.layer[l].x_mlp_norm, G.layer[l].gate, G.layer[l].up,
                    G.layer[l].gate_act, G.layer[l].prod, G.layer[l].mlp_out);
  }

  // Tail: norm + logits (append one more mbuf for Xn)
  Tid Xn = alloc_tensor(P);
  add_rmsnorm(P, (cfg.n_layers? G.layer.back().x_next : G.X_stream), W.w_rms_final, Xn, cfg.rms_eps);
  add_linear_qaware(P, Xn, W.lm_head_T, std::nullopt, W.lm_head_q4, G.logits);
}

} // namespace executorch::mlx
