import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List
import torch.nn.functional as F
import json
import os
from program_builder import ProgramBuilder
from typing import Optional, Tuple

class CustomRMSNorm(nn.Module):
    def __init__(self, base_rms: nn.Module):
        super().__init__()
        self.weight = base_rms.weight
        self.eps = float(
            getattr(base_rms, "eps", getattr(base_rms, "variance_epsilon", 1e-5))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.mlx.rms_norm(x, self.weight, self.eps)
        return out

def kv_update_and_window_checked_inplace(
    k_cache: torch.Tensor,  # [B, Hkv, T_max, D]
    v_cache: torch.Tensor,  # [B, Hkv, T_max, D]
    k_step: torch.Tensor,   # [B, Hkv, T_step, D]
    v_step: torch.Tensor,   # [B, Hkv, T_step, D]
    input_pos: int,
):
    B, Hkv, T_max, D = k_cache.shape
    _, _, T_step, _ = k_step.shape

    # writable = min(T_step, T_max - input_pos)
    writable = T_step
    if writable > 0:
        k_cache[:, :, input_pos:input_pos + writable, :].copy_(k_step[:, :, :writable, :])
        v_cache[:, :, input_pos:input_pos + writable, :].copy_(v_step[:, :, :writable, :])

    end = input_pos + T_step
    # end = min(end, T_max)
    k_win = k_cache[:, :, 0:end, :]
    v_win = v_cache[:, :, 0:end, :]
    return k_win, v_win

def _get_attr_any(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _infer_heads_dims(
    attn_module: nn.Module,
    fallback_hidden_size: int,
    fallback_num_heads: int,
    fallback_num_kv_heads: int,
):
    q_proj = _get_attr_any(attn_module, "q_proj")
    hidden_size = None
    if q_proj is not None and hasattr(q_proj, "out_features"):
        try:
            hidden_size = int(q_proj.out_features)
        except Exception:
            hidden_size = None
    if hidden_size is None:
        hidden_size = int(
            _get_attr_any(attn_module, "hidden_size", default=fallback_hidden_size)
        )

    num_heads = _get_attr_any(attn_module, "num_heads")
    if num_heads is None:
        num_heads = fallback_num_heads
    num_heads = int(num_heads)

    num_kv_heads = _get_attr_any(attn_module, "num_key_value_heads", "n_kv_heads")
    if num_kv_heads is None:
        num_kv_heads = fallback_num_kv_heads
    num_kv_heads = int(num_kv_heads)

    head_dim = _get_attr_any(attn_module, "head_dim")
    if head_dim is None:
        head_dim = hidden_size // max(1, num_heads)
    head_dim = int(head_dim)
    return hidden_size, num_heads, num_kv_heads, head_dim

class KVCacheAttention(nn.Module):
    def __init__(
        self,
        attn_module: nn.Module,
        *,
        fallback_hidden_size: int,
        fallback_num_heads: int,
        fallback_num_kv_heads: int,
        time_axis: int = 1,
        T_max: int = 4096,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.q_proj = _get_attr_any(attn_module, "q_proj")
        self.k_proj = _get_attr_any(attn_module, "k_proj")
        self.v_proj = _get_attr_any(attn_module, "v_proj")
        self.o_proj = _get_attr_any(attn_module, "o_proj", "out_proj", "o_proj_linear")
        if any(x is None for x in (self.q_proj, self.k_proj, self.v_proj, self.o_proj)):
            raise AttributeError(
                "Attention module missing q_proj/k_proj/v_proj/o_proj(out_proj)"
            )

        hidden_size, H, Hkv, Dh = _infer_heads_dims(
            attn_module,
            fallback_hidden_size,
            fallback_num_heads,
            fallback_num_kv_heads,
        )
        self.hidden_size = hidden_size
        self.num_heads = H  # Q heads
        self.num_key_value_heads = Hkv
        self.head_dim = Dh
        self.time_axis = int(time_axis)
        self.T_max = int(T_max)
        self.is_causal = True

        k0 = torch.zeros((1, self.num_key_value_heads, self.T_max, self.head_dim), dtype=dtype)
        v0 = torch.zeros((1, self.num_key_value_heads, self.T_max, self.head_dim), dtype=dtype)
        self.register_buffer("k_cache", k0, persistent=False)
        self.register_buffer("v_cache", v0, persistent=False)

    def forward(self, hidden_states: torch.Tensor, input_pos: int):
        torch._check(hidden_states.size(0) == 1)
        B, T, _ = hidden_states.shape
        H, Hkv, Dh = self.num_heads, self.num_key_value_heads, self.head_dim

        # 1) projections
        q_lin = self.q_proj(hidden_states)  # [B,T,H*D]
        k_lin = self.k_proj(hidden_states)  # [B,T,Hkv*D]
        v_lin = self.v_proj(hidden_states)  # [B,T,Hkv*D]

        # 2) reshape to [B,T,H,D] / [B,T,Hkv,D]
        q_bthd = q_lin.view(B, T, H, Dh)
        k_bthd = k_lin.view(B, T, Hkv, Dh)
        v_bthd = v_lin.view(B, T, Hkv, Dh)

        # 3) permute to B,H,T,D because rope + sdpa want that
        q_bhtd = q_bthd.permute(0, 2, 1, 3).contiguous()     # [B,H,T,D]
        k_bhtd = k_bthd.permute(0, 2, 1, 3).contiguous()     # [B,Hkv,T,D]
        v_bhtd = v_bthd.permute(0, 2, 1, 3).contiguous()     # [B,Hkv,T,D]

        # 4) RoPE on B,H,T,D (this matches your custom op signature)
        q_bhtd, k_bhtd = torch.ops.mlx.apply_rope(
            q_bhtd,               # [B,H,T,D]
            k_bhtd,               # [B,Hkv,T,D]
            self.head_dim,
            input_pos,
            traditional=False,
            base=500000.0,
            scale=1.0,
            freqs=None,
        )

        # 5) update KV cache (now both in B,Hkv,T,D)
        k_win, v_win = kv_update_and_window_checked_inplace(
            self.k_cache,
            self.v_cache,
            k_bhtd,
            v_bhtd,
            input_pos,
        )

        # 6) SDPA in B,H,T,D
        q_ = q_bhtd                    # [B,H,T,D]
        k_ = k_win                     # [B,Hkv,T,D]
        v_ = v_win                     # [B,Hkv,T,D]

        B_, Hq_, T_, Dh_ = q_.shape
        _, Hkv_, Tk_, Dhk_ = k_.shape
        assert Dh_ == Dhk_

        if Hq_ != Hkv_:
            torch._check(Hq_ >= Hkv_)
            torch._check(Hq_ % Hkv_ == 0)
            group = Hq_ // Hkv_
            k_ = k_.repeat_interleave(group, dim=1)
            v_ = v_.repeat_interleave(group, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_,  # [B,H,T,D]
            k_,
            v_,
            attn_mask=None,
            is_causal=True,
            scale=None,
        )  # → [B,H,T,D]

        # 7) back to [B,T,H*D] → out proj
        attn_out = (
            attn_out.permute(0, 2, 1, 3)   # [B,T,H,D]
            .contiguous()
            .view(B, T, H * Dh)
        )
        out = self.o_proj(attn_out)
        return out

class LlamaWithFunctionalKV(nn.Module):
    def __init__(self, base: AutoModelForCausalLM, time_axis: int = 1):
        super().__init__()
        self.model = base

        # swap rms norms
        for layer in self.model.model.layers:
            layer.input_layernorm = CustomRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = CustomRMSNorm(layer.post_attention_layernorm)
        self.model.model.norm = CustomRMSNorm(self.model.model.norm)

        cfg = base.config
        fallback_hidden_size = int(getattr(cfg, "hidden_size"))
        fallback_num_heads = int(getattr(cfg, "num_attention_heads"))
        fallback_num_kv_heads = int(getattr(cfg, "num_key_value_heads", fallback_num_heads))
        T_max = 4096 # int(getattr(cfg, "max_position_embeddings", 4096))
        dtype = base.model.embed_tokens.weight.dtype

        # wrap attention modules
        for layer in self.model.model.layers:
            layer.self_attn = KVCacheAttention(
                layer.self_attn,
                fallback_hidden_size=fallback_hidden_size,
                fallback_num_heads=fallback_num_heads,
                fallback_num_kv_heads=fallback_num_kv_heads,
                time_axis=time_axis,
                T_max=T_max,
                dtype=dtype,
            )

    def forward(self, token_ids: torch.Tensor, input_pos: int):
        m = self.model
        hs = m.model.embed_tokens(token_ids)
        for layer in m.model.layers:
            residual = hs
            hs = layer.input_layernorm(hs)
            hs = residual + layer.self_attn(hs, input_pos)
            residual = hs
            hs = layer.post_attention_layernorm(hs)
            hs = layer.mlp(hs)
            hs = residual + hs
        hs = m.model.norm(hs)
        logits = m.lm_head(hs)
        return logits


# ---------------------------------------------------------------------
# helpers for eager check
# ---------------------------------------------------------------------
def read_prompt_ids(path: str) -> List[int]:
    ids: List[int] = []
    with open(path, "r") as f:
        for line in f:
            for tok in line.strip().split():
                ids.append(int(tok))
    if not ids:
        raise RuntimeError("prompt file is empty")
    return ids

# ---------------------------------------------------------------------
# main: eager + export
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id)
    model = LlamaWithFunctionalKV(base)
    model.eval()

    from torchao.quantization.quant_api import quantize_, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup
    model = model.to(torch.bfloat16)
    quantize_(model, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)), lambda m, fqn: isinstance(m, torch.nn.Embedding))
    quantize_(model, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(64)))

    # Tie embedding
    model.model.lm_head.weight = model.model.model.embed_tokens.weight

    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long).reshape(1, -1)
    input_pos = 0
    example_inputs = (token_ids, input_pos)
    dynamic_shapes = {
        "token_ids": {1: torch.export.Dim.AUTO},
        "input_pos": torch.export.Dim.AUTO,
    }
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    ep = ep.run_decompositions({})
    
    print(ep)

    P = ProgramBuilder(ep)
    prog_json = P.build()
    with open("prog.json", "w") as f:
        f.write(prog_json)
    P.save_constant_data("consts.safetensors")

    messages = [
        {"role": "user", "content": (
            "Summarize the following article in one concise sentence. "
            "Make sure to capture not only the main argument but also "
            "the author’s tone and reasoning style.\n\n"
            "Article:\n"
            "Over the last two decades, the relationship between human creativity and "
            "artificial intelligence has evolved from curiosity to collaboration. "
            "Early AI systems were limited to mimicking patterns in existing datasets, "
            "producing outputs that were statistically coherent but semantically shallow. "
            "As models grew in scale and complexity, they began to exhibit emergent "
            "behaviors that blurred the boundary between analysis and imagination. "
            "Today, we find ourselves in a moment where AI-generated writing, music, "
            "and visual art are no longer novelties—they are competing directly with "
            "human work in professional and academic spaces.\n\n"
            "The ethical implications are immense. When an algorithm generates a "
            "painting inspired by a thousand human artists, who owns the result? "
            "When a language model writes poetry that moves readers to tears, "
            "should we attribute emotional intent to the machine or to the collective "
            "data that shaped it? These questions have shifted the conversation "
            "from what AI can do to what AI should do. Researchers argue that creativity, "
            "at its core, remains a human trait rooted in subjective experience, empathy, "
            "and moral context—dimensions still absent from even the most advanced models.\n\n"
            "However, dismissing AI as a tool without agency ignores its growing influence. "
            "In education, art, and science, humans increasingly collaborate with generative "
            "systems as creative partners. The frontier is no longer about replacement but "
            "augmentation: discovering how machines can help us think differently, see "
            "differently, and imagine new possibilities. Critics warn that relying on AI "
            "too heavily risks homogenizing culture—reducing art to what is statistically "
            "most probable rather than uniquely meaningful. Supporters counter that human "
            "oversight ensures diversity, and that creativity has always been shaped by "
            "our tools, from paintbrushes to algorithms.\n\n"
            "In this evolving dialogue, the real question may not be whether AI can create, "
            "but whether humans can adapt fast enough to coexist with a new kind of intelligence—"
            "one that mirrors us, amplifies us, and challenges us to redefine what it means "
            "to be original."
        )}
    ]

    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_file = "prompt_ids.txt"
    with open(prompt_file, "w") as f:
        f.write(" ".join(str(i) for i in ids))
