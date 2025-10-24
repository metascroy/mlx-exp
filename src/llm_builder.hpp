// llm_builder.hpp — sequential mbuf allocation (no constant/mbuf overlap)
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

#include <torch/serialize.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace executorch::mlx {

inline constexpr bool    kUseQuantMatmul = true;
inline constexpr bool    kUseQuantEmbed  = true;
inline constexpr DTypeId kComputeDT      = DTypeId::f32;

static inline ::mlx::core::Dtype to_mlx(DTypeId d) {
  using namespace ::mlx::core;
  switch (d) {
    case DTypeId::f16:  return float16;
    case DTypeId::bf16: return bfloat16;
    default:            return float32;
  }
}

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
  std::optional<Vid> w_q4, scales, biases; // biases = quant affine term
  int         group_size{64};
  std::string mode{"affine"};
  DTypeId     out_dtype{kComputeDT};
  bool valid() const { return w_q4.has_value() && scales.has_value(); }
};
struct Q4Embedding {
  std::optional<Vid> table_q4, scales, biases;
  int         group_size{64};
  std::string mode{"affine"};
  DTypeId     out_dtype{kComputeDT};
  bool valid() const { return table_q4.has_value() && scales.has_value(); }
};

struct LlamaLayerWeights {
  Vid w_rms_attn, w_rms_mlp;
  Vid Wq, Wk, Wv, Wo;          // stored as [O, I]
  Vid W_gate, W_up, W_down;    // stored as [O, I]
  Q4Linear Wq_q4, Wk_q4, Wv_q4, Wo_q4, W_gate_q4, W_up_q4, W_down_q4;
};
struct LlamaWeights {
  Vid         tok_emb;     // [vocab, Dm]
  Q4Embedding tok_emb_q4;
  Vid      lm_head_T;      // [O, I] = [vocab, Dm]
  Q4Linear lm_head_q4;
  Vid w_rms_final;
  std::vector<LlamaLayerWeights> layers;
};

// ---------- Small op helpers ----------
inline void add_rmsnorm(Program& P, Vid x, Vid w, Vid out, float eps) {
  RMSNormNode n; n.x=x; n.weight=w; n.out=out; n.eps=eps;
  P.code.push_back(make_RMS_NORM(std::move(n)));
}

inline void add_linear_qaware(Program& P, Vid x, Vid W, std::optional<Vid> b,
                              const Q4Linear& q, Vid out) {
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

inline void add_project_pack_heads(Program& P, Vid x, Vid W, std::optional<Vid> b,
                                   int B,int Hout,int Dh, Vid outBHTDh, const Q4Linear& qmeta) {
  Vid tmp = outBHTDh;
  add_linear_qaware(P, x, W, b, qmeta, tmp);
  ReshapeNode r; r.x=tmp; r.out=tmp; r.shape={B,-1,Hout,Dh}; P.code.push_back(make_RESHAPE(std::move(r)));
  TransposeNode t; t.x=tmp; t.out=outBHTDh; t.perm={0,2,1,3}; P.code.push_back(make_TRANSPOSE(std::move(t)));
}

inline void add_rope_qk(Program& P, Vid q_in, Vid k_in, int Dh, Vid pos_scalar,
                        Vid q_out, Vid k_out, bool traditional, float theta, float scale=1.f) {
  RopeNode rn; rn.q_in=q_in; rn.k_in=k_in; rn.q_out=q_out; rn.k_out=k_out;
  rn.head_dim=Dh; rn.traditional=traditional; rn.base=theta; rn.scale=scale; rn.pos=pos_scalar;
  P.code.push_back(make_ROPE_APPLY(std::move(rn)));
}

inline void add_sdpa(Program& P, Vid q, Vid k, Vid v, float scale,
                     std::optional<Vid> mask, Vid out, bool causal) {
  SdpaNode n; n.q=q; n.k=k; n.v=v; n.out=out; n.scale=scale; n.mask=mask; n.causal=causal;
  P.code.push_back(make_SDPA(std::move(n)));
}

inline void add_kv_write_read(Program& P, const struct CacheIds& c, Vid K_step, Vid V_step, Vid K_win, Vid V_win);

// ---------- One layer ----------
inline void add_llama_layer(Program& P, const LlamaCfg& cfg, const LlamaLayerWeights& W,
                            const struct CacheIds& Cc, Vid pos_scalar,
                            Vid x_in_BTDm, Vid x_out_BTDm,
                            Vid x_norm, Vid Q, Vid K, Vid V,
                            Vid Qr, Vid Kr, Vid Kwin, Vid Vwin,
                            Vid Attn_BHTDh, Vid Attn_BTDm,
                            Vid x_res1, Vid x_mlp_norm, Vid gate, Vid up, Vid gate_act, Vid prod, Vid mlp_out) {
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

// ---------- Loader (constants only) ----------
inline LlamaWeights load_llama_weights_from_torch(const std::string& path,
                                                  Program& P,
                                                  const LlamaCfg& cfg,
                                                  bool tie_lm_head=true) {
  using namespace ::mlx::core;

  auto to_shape = [](at::IntArrayRef s){ std::vector<int> v(s.begin(), s.end()); return Shape(v.begin(), v.end()); };
  auto to_f32_owned = [&](const torch::Tensor& tin, bool tpose=false)->array {
    torch::Tensor t = tin.to(torch::kCPU);
    if (t.scalar_type()!=torch::kFloat32) t = t.to(torch::kFloat32);
    t = t.contiguous();
    array a(t.data_ptr<float>(), to_shape(t.sizes()), float32);
    if (tpose && a.ndim()==2) a = contiguous(transpose(a,{1,0}));
    a = contiguous(a);
    if (kComputeDT!=DTypeId::f32) a = astype(a, to_mlx(kComputeDT));
    return a;
  };
  auto push_const = [&](array a)->Vid {
    Vid id{ P.num_constants++ };
    if (P.C_tensors.size() < P.num_constants) P.C_tensors.resize(P.num_constants);
    if (P.meta.size() < P.total_slots()) P.meta.resize(P.total_slots());
    P.meta[id.idx].kind = SlotKind::Tensor;
    P.set_constant_tensor(id, std::move(a));
    return id;
  };

  // read file
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("cannot open model: "+path);
  ifs.seekg(0,std::ios::end); auto n=ifs.tellg(); ifs.seekg(0);
  std::vector<char> bytes((size_t)n);
  if (!ifs.read(bytes.data(), n)) throw std::runtime_error("read failed: "+path);

  at::IValue iv = torch::pickle_load(bytes);
  if (!iv.isGenericDict()) throw std::runtime_error("pickle root not dict");
  auto dict = iv.toGenericDict();
  if (dict.contains("state_dict")) dict = dict.at("state_dict").toGenericDict();

  auto has = [&](const char* k){ return dict.contains(k); };
  auto get = [&](const char* k)->torch::Tensor {
    if (!dict.contains(k)) throw std::runtime_error(std::string("missing key: ")+k);
    return dict.at(k).toTensor();
  };

  LlamaWeights W;

  const char* k_embed = has("model.embed_tokens.weight") ? "model.embed_tokens.weight" :
                        (has("tok_embeddings.weight") ? "tok_embeddings.weight" : "model.embed_tokens.weight");
  W.tok_emb = push_const(to_f32_owned(get(k_embed), /*tpose=*/false));  // [vocab, Dm]

  const char* kfn = has("model.norm.weight") ? "model.norm.weight" :
                    (has("norm.weight") ? "norm.weight" : "model.norm.weight");
  W.w_rms_final = push_const(to_f32_owned(get(kfn), false));

  W.layers.resize(cfg.n_layers);
  for (int l=0;l<cfg.n_layers;++l) {
    auto key = [&](const std::string& s){ return "model.layers."+std::to_string(l)+"."+s; };
    auto& Lw = W.layers[l];
    Lw.w_rms_attn = push_const(to_f32_owned(get(key("input_layernorm.weight").c_str()), false));
    Lw.w_rms_mlp  = push_const(to_f32_owned(get(key("post_attention_layernorm.weight").c_str()), false));
    // Store weights as [O, I] now (no transpose) — interpreter does W^T
    Lw.Wq     = push_const(to_f32_owned(get(key("self_attn.q_proj.weight").c_str()), false));
    Lw.Wk     = push_const(to_f32_owned(get(key("self_attn.k_proj.weight").c_str()), false));
    Lw.Wv     = push_const(to_f32_owned(get(key("self_attn.v_proj.weight").c_str()), false));
    Lw.Wo     = push_const(to_f32_owned(get(key("self_attn.o_proj.weight").c_str()), false));
    Lw.W_gate = push_const(to_f32_owned(get(key("mlp.gate_proj.weight").c_str()), false));
    Lw.W_up   = push_const(to_f32_owned(get(key("mlp.up_proj.weight").c_str()),   false));
    Lw.W_down = push_const(to_f32_owned(get(key("mlp.down_proj.weight").c_str()), false));
  }

  const bool have_lm_head = has("lm_head.weight") || has("model.lm_head.weight");
  if (!tie_lm_head && have_lm_head) {
    const char* k_lm = has("lm_head.weight") ? "lm_head.weight" : "model.lm_head.weight";
    // Store [O, I] (no transpose). Interpreter transposes at runtime.
    W.lm_head_T = push_const(to_f32_owned(get(k_lm), /*tpose=*/false));
  } else {
    // Tie to embeddings: reuse [vocab, Dm] directly (no transpose).
    using namespace ::mlx::core;
    array LT = P.C_tensors[W.tok_emb.idx].value();
    if (kComputeDT!=DTypeId::f32) LT = astype(LT, to_mlx(kComputeDT));
    W.lm_head_T = push_const(contiguous(LT));
  }

  // -------- Quantization (FP weights are [O, I], interpreter uses transpose=true) --------
  const bool will_share_embed_q = (tie_lm_head && kUseQuantEmbed);

  if (kUseQuantMatmul) {
    using namespace ::mlx::core;

    // Quantize a [O, I] weight as-is for quantized matmul with transpose=true
    auto qlin = [&](Vid src, Q4Linear& out){
      array Woi = P.C_tensors[src.idx].value();        // [O, I]
      auto q = quantize(Woi, 64, 4, "affine", {});     // packs along feature dim
      out.w_q4   = push_const(contiguous(q[0]));
      out.scales = push_const(contiguous(q[1]));
      if (q.size()>=3) out.biases = push_const(contiguous(q[2]));
      out.group_size=64; out.mode="affine"; out.out_dtype=kComputeDT;
    };

    // Layer weights
    for (auto& Lw : W.layers) {
      qlin(Lw.Wq,     Lw.Wq_q4);
      qlin(Lw.Wk,     Lw.Wk_q4);
      qlin(Lw.Wv,     Lw.Wv_q4);
      qlin(Lw.Wo,     Lw.Wo_q4);
      qlin(Lw.W_gate, Lw.W_gate_q4);
      qlin(Lw.W_up,   Lw.W_up_q4);
      qlin(Lw.W_down, Lw.W_down_q4);
    }

    // lm_head: if we are *not* going to share embed packs, quantize it now
    if (!will_share_embed_q) {
      qlin(W.lm_head_T, W.lm_head_q4);
    }
  }

  if (kUseQuantEmbed) {
    using namespace ::mlx::core;
    // Quantize the embedding table [vocab, Dm] as-is.
    auto q = quantize(P.C_tensors[W.tok_emb.idx].value(), 64, 4, "affine", {});
    W.tok_emb_q4.table_q4 = push_const(contiguous(q[0]));
    W.tok_emb_q4.scales   = push_const(contiguous(q[1]));
    if (q.size()>=3) W.tok_emb_q4.biases = push_const(contiguous(q[2]));
    W.tok_emb_q4.group_size=64; W.tok_emb_q4.mode="affine"; W.tok_emb_q4.out_dtype=kComputeDT;

    // If tying AND matmul quant is enabled: reuse embed quant packs for lm_head
    if (tie_lm_head && kUseQuantMatmul) {
      W.lm_head_q4.w_q4      = W.tok_emb_q4.table_q4;
      W.lm_head_q4.scales    = W.tok_emb_q4.scales;
      W.lm_head_q4.biases    = W.tok_emb_q4.biases; // may be nullopt — ok
      W.lm_head_q4.group_size= W.tok_emb_q4.group_size;
      W.lm_head_q4.mode      = W.tok_emb_q4.mode;
      W.lm_head_q4.out_dtype = W.tok_emb_q4.out_dtype;
    }
  }

  return W;
}

// ---------- Cache ids (append mbufs sequentially) ----------
struct CacheIds {
  Vid K_cache, V_cache;
  Vid cursor;        // Scalar (position)
  // Scalar Vids for slice write window
  Vid sh_axis, sh_start, sh_len;
  // Scalar Vids for slice read window
  Vid rd_axis, rd_start, rd_len;
};
struct LlamaCaches { std::vector<CacheIds> by_layer; };

// appends one mbuf, returns Vid
inline Vid _alloc_mbuf(Program& P) {
  Vid id{ static_cast<uint32_t>(P.mbufs_begin() + P.num_mutable_buffers) };
  ++P.num_mutable_buffers;
  if (P.meta.size() <= id.idx) P.meta.resize(id.idx+1);
  P.meta[id.idx].kind = SlotKind::Tensor; // default; builder may change to Scalar
  return id;
}

inline LlamaCaches make_caches(Program& P, const LlamaCfg& cfg) {
  LlamaCaches C; C.by_layer.resize(cfg.n_layers);

  // shared helpers
  Vid cursor = _alloc_mbuf(P);
  P.meta[cursor.idx].kind = SlotKind::Scalar;

  // Write window scalars
  Vid sh_axis  = _alloc_mbuf(P); P.meta[sh_axis.idx].kind  = SlotKind::Scalar;
  Vid sh_start = _alloc_mbuf(P); P.meta[sh_start.idx].kind = SlotKind::Scalar;
  Vid sh_len   = _alloc_mbuf(P); P.meta[sh_len.idx].kind   = SlotKind::Scalar;

  // Read window scalars
  Vid rd_axis  = _alloc_mbuf(P); P.meta[rd_axis.idx].kind  = SlotKind::Scalar;
  Vid rd_start = _alloc_mbuf(P); P.meta[rd_start.idx].kind = SlotKind::Scalar;
  Vid rd_len   = _alloc_mbuf(P); P.meta[rd_len.idx].kind   = SlotKind::Scalar;

  for (int l=0;l<cfg.n_layers;++l) {
    Vid Kc = _alloc_mbuf(P);
    Vid Vc = _alloc_mbuf(P);
    C.by_layer[l] = { Kc, Vc, cursor, sh_axis, sh_start, sh_len, rd_axis, rd_start, rd_len };
  }
  return C;
}

// ---------- Graph scratch (append mbufs sequentially) ----------
struct LlamaGraphIds {
  Vid input_ids;  // inputs range
  Vid logits;     // outputs range
  Vid X_embed, X_stream;
  struct LayerScratch {
    Vid x_norm, Q, K, V, Qr, Kr, Kwin, Vwin, Attn_BHTDh, Attn_BTDm,
        x_res1, x_mlp_norm, gate, up, gate_act, prod, mlp_out, x_next;
  };
  std::vector<LayerScratch> layer;
};

inline LlamaGraphIds make_graph_ids(Program& P, const LlamaCfg& cfg) {
  LlamaGraphIds G;
  G.input_ids = Vid{ P.inputs_begin() };
  G.logits    = Vid{ P.outputs_begin() };
  if (P.meta.size() < P.total_slots()) P.meta.resize(P.total_slots());
  P.meta[G.input_ids.idx].kind = SlotKind::Tensor;
  P.meta[G.logits.idx].kind    = SlotKind::Tensor;

  auto M = [&](Vid& v){ v = _alloc_mbuf(P); };

  M(G.X_embed);
  M(G.X_stream);

  G.layer.resize(cfg.n_layers);
  for (int l=0;l<cfg.n_layers;++l) {
    auto& S = G.layer[l];
    M(S.x_norm); M(S.Q); M(S.K); M(S.V);
    M(S.Qr); M(S.Kr); M(S.Kwin); M(S.Vwin);
    M(S.Attn_BHTDh); M(S.Attn_BTDm);
    M(S.x_res1); M(S.x_mlp_norm); M(S.gate); M(S.up);
    M(S.gate_act); M(S.prod); M(S.mlp_out);
    M(S.x_next);
  }
  return G;
}

// ---------- Build graph ----------
inline void add_kv_write_read(Program& P, const CacheIds& c, Vid K_step, Vid V_step, Vid K_win, Vid V_win) {
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
    const Vid pos_scalar = C.cursor;
    add_llama_layer(P, cfg, W.layers[l], C, pos_scalar,
                    (l==0)?G.X_stream:G.layer[l-1].x_next, G.layer[l].x_next,
                    G.layer[l].x_norm, G.layer[l].Q, G.layer[l].K, G.layer[l].V,
                    G.layer[l].Qr, G.layer[l].Kr, G.layer[l].Kwin, G.layer[l].Vwin,
                    G.layer[l].Attn_BHTDh, G.layer[l].Attn_BTDm,
                    G.layer[l].x_res1, G.layer[l].x_mlp_norm, G.layer[l].gate, G.layer[l].up,
                    G.layer[l].gate_act, G.layer[l].prod, G.layer[l].mlp_out);
  }

  // Tail: norm + logits (append one more mbuf for Xn)
  Vid Xn = _alloc_mbuf(P);
  add_rmsnorm(P, (cfg.n_layers? G.layer.back().x_next : G.X_stream), W.w_rms_final, Xn, cfg.rms_eps);
  add_linear_qaware(P, Xn, W.lm_head_T, std::nullopt, W.lm_head_q4, G.logits);
}

} // namespace executorch::mlx
