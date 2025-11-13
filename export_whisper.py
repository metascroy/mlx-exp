# whisper_export.py
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend
from transformers import AutoProcessor, WhisperForConditionalGeneration
from program_builder import ProgramBuilder

from datasets import load_dataset, Audio as HFAudio
import soundfile as sf
import numpy as np

# ---------------------------------------------------------------------
# Shared helpers (ported from your Llama example)
# ---------------------------------------------------------------------


def kv_update_and_window_checked_inplace(
    k_cache: torch.Tensor,  # [B, Hkv, T_max, D]
    v_cache: torch.Tensor,  # [B, Hkv, T_max, D]
    k_step: torch.Tensor,   # [B, Hkv, T_step, D]
    v_step: torch.Tensor,   # [B, Hkv, T_step, D]
    input_pos: int,
):
    B, Hkv, T_max, D = k_cache.shape
    _, _, T_step, _ = k_step.shape

    writable = T_step
    if writable > 0:
        k_cache[:, :, input_pos:input_pos + writable, :].copy_(
            k_step[:, :, :writable, :]
        )
        v_cache[:, :, input_pos:input_pos + writable, :].copy_(
            v_step[:, :, :writable, :]
        )

    end = input_pos + T_step
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


# ---------------------------------------------------------------------
# Encoder wrapper (audio -> hidden states)
# ---------------------------------------------------------------------


class WhisperEncoderExportable(nn.Module):
    """
    Wraps Whisper encoder for torch.export.
    Input:  input_features (B, feature_size, nb_max_frames)
    Output: last_hidden_state (B, T_enc, H)
    """

    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.get_encoder()
        self.config = self.encoder.config
        self.model_device = model.device
        self.model_dtype = model.dtype

    def forward(self, input_features: torch.FloatTensor):
        return self.encoder(input_features=input_features).last_hidden_state


# ---------------------------------------------------------------------
# KV-cached self-attention for Whisper decoder (buffer-based KV)
# ---------------------------------------------------------------------


class WhisperKVCacheAttention(nn.Module):
    """
    Whisper self-attention with an internal KV cache as module buffers.

    - Takes hidden_states (B, T, D) and input_pos (int)
    - Uses q_proj/k_proj/v_proj/o_proj from the original Whisper self_attn module
    - Maintains k_cache/v_cache buffers of shape (1, Hkv, T_max, D_head)
    - Updates cache via kv_update_and_window_checked_inplace (narrow + copy_)
    - Does scaled dot-product attention with is_causal=True
    """

    def __init__(
        self,
        attn_module: nn.Module,
        *,
        fallback_hidden_size: int,
        fallback_num_heads: int,
        fallback_num_kv_heads: int,
        T_max: int = 4096,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.q_proj = _get_attr_any(attn_module, "q_proj")
        self.k_proj = _get_attr_any(attn_module, "k_proj")
        self.v_proj = _get_attr_any(attn_module, "v_proj")
        self.o_proj = _get_attr_any(attn_module, "out_proj", "o_proj", "o_proj_linear")
        if any(x is None for x in (self.q_proj, self.k_proj, self.v_proj, self.o_proj)):
            raise AttributeError(
                "Whisper attention module missing q_proj/k_proj/v_proj/out_proj"
            )

        hidden_size, H, Hkv, Dh = _infer_heads_dims(
            attn_module,
            fallback_hidden_size,
            fallback_num_heads,
            fallback_num_kv_heads,
        )
        self.hidden_size = hidden_size
        self.num_heads = H
        self.num_key_value_heads = Hkv
        self.head_dim = Dh
        self.T_max = int(T_max)

        k0 = torch.zeros(
            (1, self.num_key_value_heads, self.T_max, self.head_dim), dtype=dtype
        )
        v0 = torch.zeros(
            (1, self.num_key_value_heads, self.T_max, self.head_dim), dtype=dtype
        )
        self.register_buffer("k_cache", k0, persistent=False)
        self.register_buffer("v_cache", v0, persistent=False)

    def forward(self, hidden_states: torch.Tensor, input_pos: int):
        torch._check(hidden_states.size(0) == 1)
        B, T, _ = hidden_states.shape
        H = self.num_heads
        Hkv = self.num_key_value_heads
        Dh = self.head_dim

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q_bthd = q_lin.view(B, T, H, Dh)
        k_bthd = k_lin.view(B, T, Hkv, Dh)
        v_bthd = v_lin.view(B, T, Hkv, Dh)

        q_bhtd = q_bthd.permute(0, 2, 1, 3).contiguous().clone()
        k_bhtd = k_bthd.permute(0, 2, 1, 3).contiguous().clone()
        v_bhtd = v_bthd.permute(0, 2, 1, 3).contiguous().clone()

        k_win, v_win = kv_update_and_window_checked_inplace(
            self.k_cache,
            self.v_cache,
            k_bhtd,
            v_bhtd,
            input_pos,
        )

        q_ = q_bhtd
        k_ = k_win
        v_ = v_win

        B_, Hq_, Tq_, Dh_ = q_.shape
        _, Hkv_, Tk_, Dhk_ = k_.shape
        assert Dh_ == Dhk_

        if Hq_ != Hkv_:
            torch._check(Hq_ >= Hkv_)
            torch._check(Hq_ % Hkv_ == 0)
            group = Hq_ // Hkv_
            k_ = k_.repeat_interleave(group, dim=1)
            v_ = v_.repeat_interleave(group, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=None,
            is_causal=True,
            scale=None,
        )

        attn_out = (
            attn_out.permute(0, 2, 1, 3)
            .contiguous()
            .view(B, T, H * Dh)
        )
        out = self.o_proj(attn_out)
        return out


# ---------------------------------------------------------------------
# Decoder with functional KV cache wired into Whisper decoder
# ---------------------------------------------------------------------


class WhisperDecoderWithFunctionalKV(nn.Module):
    """
    Forward signature:
        forward(decoder_input_ids, encoder_hidden_states, input_pos: int)
    """

    def __init__(
        self,
        base: WhisperForConditionalGeneration,
        *,
        T_max: int = 4096,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.config = base.config
        self.decoder = base.model.decoder
        self.proj_out = base.proj_out

        cfg = self.config
        fallback_hidden_size = int(getattr(cfg, "d_model"))
        fallback_num_heads = int(getattr(cfg, "decoder_attention_heads"))
        fallback_num_kv_heads = int(getattr(cfg, "decoder_attention_heads"))

        for layer in self.decoder.layers:
            layer.self_attn = WhisperKVCacheAttention(
                layer.self_attn,
                fallback_hidden_size=fallback_hidden_size,
                fallback_num_heads=fallback_num_heads,
                fallback_num_kv_heads=fallback_num_kv_heads,
                T_max=T_max,
                dtype=dtype,
            )

    def forward(
        self,
        decoder_input_ids: torch.Tensor,       # [B,T_dec]
        encoder_hidden_states: torch.Tensor,   # [B,T_enc,D]
        input_pos: int,
    ):
        decoder = self.decoder

        hs = decoder.embed_tokens(decoder_input_ids)
        if hasattr(decoder, "embed_positions"):
            pos = decoder.embed_positions(decoder_input_ids)
            hs = hs + pos

        for layer in decoder.layers:
            residual = hs
            ln = _get_attr_any(
                layer,
                "self_attn_layer_norm",
                "self_attn_layernorm",
                default=None,
            )
            if ln is None:
                raise AttributeError("Decoder layer missing self_attn_layer_norm")
            hs = ln(hs)
            hs = residual + layer.self_attn(hs, input_pos)

            if hasattr(layer, "encoder_attn"):
                residual = hs
                enc_ln = _get_attr_any(
                    layer,
                    "encoder_attn_layer_norm",
                    "encoder_attn_layernorm",
                    default=None,
                )
                if enc_ln is None:
                    raise AttributeError(
                        "Decoder layer missing encoder_attn_layer_norm"
                    )
                hs = enc_ln(hs)
                enc_attn_out = layer.encoder_attn(
                    hs,
                    encoder_hidden_states,
                    output_attentions=False,
                )[0]
                hs = residual + enc_attn_out

            residual = hs
            final_ln = _get_attr_any(layer, "final_layer_norm", default=None)
            if final_ln is None:
                raise AttributeError("Decoder layer missing final_layer_norm")
            hs = final_ln(hs)
            fc1 = _get_attr_any(layer, "fc1", "dense_relu_then_dense", default=None)
            fc2 = _get_attr_any(layer, "fc2", default=None)
            if fc1 is None or fc2 is None:
                raise AttributeError("Decoder layer missing fc1/fc2 MLP modules")
            hs = fc2(F.gelu(fc1(hs)))
            hs = residual + hs

        if hasattr(decoder, "layer_norm"):
            hs = decoder.layer_norm(hs)

        logits = self.proj_out(hs)
        return logits


# ---------------------------------------------------------------------
# Export both parts (encoder + functional-KV decoder)
# ---------------------------------------------------------------------

@torch.no_grad()
def export_whisper_encoder_decoder(
    model_id: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_hidden_seq_len: int = 4096,   # unused by Whisper encoder, kept for API parity
    max_dec_seq_len: int = 1024,      # T_max for KV cache
    batch_size: int = 1,
    strict: bool = True,
) -> Dict[str, ExportedProgram]:
    """
    Returns:
      {
        "encoder": ExportedProgram,      # (input_features) -> last_hidden_state
        "text_decoder": ExportedProgram, # (decoder_input_ids, encoder_hidden_states, input_pos) -> logits
      }
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # ------------------------------------------------------------
    # 1) Load REAL human audio from distil-whisper/librispeech_long,
    #    but avoid torchcodec by disabling HF audio decoding.
    # ------------------------------------------------------------
    dataset = load_dataset(
        "distil-whisper/librispeech_long",
        "clean",
        split="validation",
    )
    # Prevent datasets from decoding audio via torchcodec
    sample = dataset[0]["audio"]
    input_features = processor(
        sample["array"],
        return_tensors="pt",
        truncation=False,
        sampling_rate=sample["sampling_rate"],
    ).input_features
    # Current implementation of the transcibe method accepts up to 30 seconds of audio, therefore I trim the audio here.
    input_features_trimmed = input_features[:, :, :3000].contiguous()

    # For C++: save a float32 copy to disk
    inp_np = input_features_trimmed.numpy().astype("float32")
    np.savetxt(
        "whisper_encoder_input_shape.txt",
        np.array(inp_np.shape, dtype=np.int64)[None, :],
        fmt="%d",
    )
    inp_np.tofile("whisper_encoder_input.bin")  # row-major float32

    # For export: use model dtype on the target device
    input_features_export = input_features_trimmed.to(device=device, dtype=model.dtype)

    # ------------------------------------------------------------
    # 4) Export encoder
    # ------------------------------------------------------------
    encoder_wrapper = WhisperEncoderExportable(model).to(model.device).eval()
    encoder_ep = torch.export.export(
        encoder_wrapper,
        args=(input_features_export,),
        kwargs={},
        dynamic_shapes=None,
        strict=strict,
    )
    encoder_ep = encoder_ep.run_decompositions({})

    # Compute example encoder_hidden_states using the exported module
    encoder_hidden_states = encoder_ep.module()(input_features_export)  # [B,T_enc,D]

    # ------------------------------------------------------------
    # 5) Build decoder with functional KV
    # ------------------------------------------------------------
    decoder_model = WhisperDecoderWithFunctionalKV(
        base=model,
        T_max=max_dec_seq_len,
        dtype=model.dtype,
    ).to(model.device).eval()

    # Quantize decoder model
    from torchao.quantization.quant_api import quantize_, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup, PerAxis
    decoder_model = decoder_model.to(torch.bfloat16)
    # quantize_(decoder_model, IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)), lambda m, fqn: isinstance(m, torch.nn.Embedding))
    quantize_(decoder_model, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(64)))

    # Example decoder inputs:
    start_id = model.config.decoder_start_token_id or model.config.bos_token_id
    decoder_input_ids = torch.tensor(
        [[start_id]],
        device=model.device,
        dtype=torch.long,
    )  # [1,1]

    input_pos = 3  # scalar; made symbolic via dynamic_shapes

    example_inputs = (decoder_input_ids, encoder_hidden_states, input_pos)
    print("SHAPES", decoder_input_ids.shape, encoder_hidden_states.shape)

    dynamic_shapes = {
        "decoder_input_ids": {},         # fixed shape for now
        "encoder_hidden_states": {},     # fixed shape for now
        "input_pos": torch.export.Dim.AUTO,
    }

    # Export decoder
    # with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
    text_decoder_ep = torch.export.export(
        decoder_model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
    )
    text_decoder_ep = text_decoder_ep.run_decompositions({})

    return {
        "encoder": encoder_ep,
        "text_decoder": text_decoder_ep,
    }


# ---------------------------------------------------------------------
# Tiny CLI helper
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_id = "openai/whisper-large-v3-turbo"
    eps = export_whisper_encoder_decoder(
        model_id=model_id,
        device="cpu",
        dtype=torch.bfloat16,
        max_dec_seq_len=1024,
        batch_size=1,
        strict=False,
    )
    print("Exported:", list(eps.keys()))
    for name, ep in eps.items():
        print("=== ", name, " ===")
        print(ep)

    for k in eps:
        P = ProgramBuilder(eps[k])
        prog_json = P.build()
        with open(f"whisper_{k}_prog.json", "w") as f:
            f.write(prog_json)
        P.save_constant_data(f"whisper_{k}_consts.safetensors")
