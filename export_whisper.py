import logging
from typing import Dict, Optional, Any, Tuple

import torch
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend
from transformers import (
    AutoProcessor,
    EncoderDecoderCache,
    StaticCache,
    WhisperForConditionalGeneration,
    CacheLayerMixin,
    Cache,
    PretrainedConfig,
)
from transformers.utils import is_torchdynamo_compiling
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

logger = logging.getLogger(__name__)

QUANTIZE = True
DEFAULT_EXPORT_KWARGS = {
    "batch_size": 1,
    "max_decoder_seq_len": 256,
    # FP32 runs faster than BF16 in MLX when not quantized, needs investigation
    "model_dtype": {True: torch.bfloat16, False: torch.float32}[QUANTIZE],
    "quantize": QUANTIZE,
}

class WhisperCacheLayer(CacheLayerMixin):
    """
    Static KV cache layer for Whisper.

    - Backing storage: [batch_size, num_heads, max_cache_len, head_dim]
    - Uses `narrow(...).copy_` for updates.
    - `cache_position` is a 1D tensor; we call `.item()` on it and
      `torch._check_is_size(start)` exactly as you specified.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len
        # Filled on lazy_initialization
        self.max_batch_size: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0
        self.dtype: torch.dtype = torch.float32
        self.device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        """
        Allocate static backing tensors using the first key_states call to
        infer batch size, num heads, head dim, dtype and device.

        key_states: (B, H, L, D)
        """
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,    # (B, H, L, D)
        value_states: torch.Tensor,  # (B, H, L, D)
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cache_kwargs may contain:
          * cache_position: a 1D tensor giving the start index in the cache
            for this block of L tokens.

        This follows your pattern exactly:
          - cache_position is a 1D tensor
          - we call `.item()` on it
          - we call `torch._check_is_size(start)`
          - we use narrow(...).copy_
        """
        # Lazy init if needed (mirrors StaticLayer behavior)
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position", None)

        # DO NOT reassign self.keys / self.values — keep aliasing with
        # any registered buffers intact.
        k_out = self.keys
        v_out = self.values

        if cache_position is None:
            # Cross-attention (or full overwrite) path.
            # Copy the whole sequence into the layer cache; we assume
            # key_states.shape[-2] <= max_cache_len.
            L = key_states.shape[-2]
            k_out.narrow(2, 0, L).copy_(key_states)
            v_out.narrow(2, 0, L).copy_(value_states)
        else:
            # Self-attention path: cache_position is the start index
            # for a contiguous block of L tokens.
            L = key_states.shape[-2]

            # cache_position is a 1D tensor; use .item() and _check_is_size
            assert isinstance(cache_position, torch.Tensor), "cache_position must be a tensor"
            torch._check(cache_position.numel() == 1)
            start = cache_position.item()
            torch._check_is_size(start)

            k_slice = k_out.narrow(2, start, L)
            v_slice = v_out.narrow(2, start, L)

            k_slice.copy_(key_states)
            v_slice.copy_(value_states)

        return k_out, v_out

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        """
        For a static cache, we can just mirror StaticLayer semantics:
          kv_offset = 0
          kv_length = max_cache_len
        """
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """
        Approximate occupied length by counting non-zero positions along
        time dim for the first (batch, head), same as StaticLayer.
        """
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return (self.keys[0, 0].any(dim=-1)).sum().item()

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class WhisperCache(Cache):
    """
    Static cache for Whisper using WhisperCacheLayer per layer.

    You can use this both for:
      - self-attention  (max_cache_len = max_decoder_seq_len)
      - cross-attention (max_cache_len = encoder_seq_len)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        decoder_config = config.get_text_config(decoder=True)

        num_layers = decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            num_layers = num_layers - decoder_config.num_kv_shared_layers

        layers: list[CacheLayerMixin] = [
            WhisperCacheLayer(max_cache_len=max_cache_len)
            for _ in range(num_layers)
        ]

        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )


# ---------------------------------------------------------------------
# Whisper encoder wrapper
# ---------------------------------------------------------------------


class WhisperEncoderExportable(torch.nn.Module):
    """
    Thin wrapper around Whisper's encoder so torch.export sees a simple
    `forward(input_features) -> encoder_hidden_states`.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.FloatTensor) -> torch.FloatTensor:
        # Whisper encoder takes `input_features` and returns a BaseModelOutput
        return self.encoder(input_features=input_features).last_hidden_state


# ---------------------------------------------------------------------
# Cross-attention projections wrapper (NEW)
# ---------------------------------------------------------------------


class WhisperCrossAttentionProjections(torch.nn.Module):
    """
    Compute *only* the cross-attention K/V projections for all decoder
    layers, given encoder_hidden_states.

    forward(
        encoder_hidden_states: (B, T_enc, H)
    ) -> (k_cache, v_cache)

    where
      k_cache: (num_layers, B, num_heads, T_enc, head_dim)
      v_cache: (num_layers, B, num_heads, T_enc, head_dim)
    """

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    @staticmethod
    def _reshape_to_heads(
        x: torch.Tensor,   # (B, T_enc, embed_dim)
        seq_len: int,
        bsz: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        # (B, T_enc, H*D) -> (B, T_enc, H, D) -> (B, H, T_enc, D)
        x = x.view(bsz, seq_len, num_heads, head_dim)
        x = x.transpose(1, 2)  # (B, H, T_enc, D)
        return x

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns two tuples of per-layer K/V tensors instead of stacked tensors.
        
        Returns:
            (k_tuple, v_tuple) where each is a tuple of num_layers tensors,
            each with shape (B, H, T_enc, D)
        """
        bsz, seq_len, _ = encoder_hidden_states.shape

        k_list = []
        v_list = []

        for layer in self.decoder.layers:
            cross_attn = layer.encoder_attn

            # Linear projections: (B, T_enc, embed_dim)
            k = cross_attn.k_proj(encoder_hidden_states)
            v = cross_attn.v_proj(encoder_hidden_states)

            num_heads = cross_attn.num_heads
            head_dim = cross_attn.head_dim

            # (B, H, T_enc, D)
            k = self._reshape_to_heads(k, seq_len, bsz, num_heads, head_dim)
            v = self._reshape_to_heads(v, seq_len, bsz, num_heads, head_dim)

            k_list.append(k)
            v_list.append(v)

        # Return as tuples (not stacked) for easier per-layer access
        return tuple(k_list), tuple(v_list)




# ---------------------------------------------------------------------
# Custom SDPA implementation (HF AttentionInterface)
# ---------------------------------------------------------------------


def whisper_sdpa_impl(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):
    """
    Custom SDPA wrapper for Whisper that:

      * Assumes `attention_mask` is already the full, precomputed mask
        (e.g., causal + padding), sliced at the caller.
      * Forces `is_causal=False` so sdpa does NOT add its own causal mask.
      * Otherwise defers to the standard HF sdpa_attention_forward, so it
        still handles things like GQA (repeat_kv) etc.

    The signature matches transformers.integrations.sdpa_attention.sdpa_attention_forward
    so it can be used with AttentionInterface.register.
    """
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        # We always treat the mask as fully precomputed:
        is_causal=False,
        # Pass through any extra kwargs (output_attentions, head_mask, etc.)
        **kwargs,
    )


def register_whisper_attention(model: WhisperForConditionalGeneration):
    impl_name = "whisper_sdpa_precomputed_mask"
    AttentionInterface.register(impl_name, whisper_sdpa_impl)
    # Tell Whisper to use this attention implementation
    model.config._attn_implementation = impl_name


# ---------------------------------------------------------------------
# Whisper decoder wrapper with static self- and cross-attn caches + mask
# ---------------------------------------------------------------------


class WhisperDecoderWithStaticCache(torch.nn.Module):
    """
    Wrapper around Whisper decoder with:
      * A static self-attention cache implemented with WhisperCache
        (using narrow + copy_ for updates).
      * A static cross-attention cache implemented with WhisperCache
        (encoder sequence length is treated as fixed).
      * Cache tensors registered as buffers on each decoder layer:
          self_attention_key_cache, self_attention_value_cache,
          cross_attention_key_cache, cross_attention_value_cache.
      * A single precomputed causal attention mask buffer on the wrapper,
        sliced at the start of forward and passed as `attention_mask`.
      * No DynamicCache anywhere.

    forward(
        decoder_input_ids: (B, T_dec),
        cross_k_cache: (L, B, H, T_enc, D),
        cross_v_cache: (L, B, H, T_enc, D),
        cache_position: int   # symbolic int, start index for this block
    ) -> (B, T_dec, vocab_size)
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        max_static_cache_length: int,
        batch_size: int,
        encoder_seq_len: int,
    ):
        super().__init__()

        self.decoder = model.get_decoder()
        # Whisper uses `proj_out` instead of `lm_head`
        self.proj_out = model.proj_out
        self.config = model.config
        self.max_cache_len = max_static_cache_length
        self.encoder_seq_len = encoder_seq_len

        device = model.device
        dtype = model.dtype

        # ---- Self-attention cache (static, length = max_decoder_seq_len) ----
        head_dim_dec = getattr(
            self.config,
            "head_dim",
            self.config.d_model // self.config.decoder_attention_heads,
        )
        num_heads_dec = getattr(
            self.config,
            "num_key_value_heads",
            self.config.decoder_attention_heads,
        )

        self.self_attention_cache = WhisperCache(
            config=self.config,
            max_cache_len=max_static_cache_length,
        )
        self.self_attention_cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_heads_dec,
            head_dim=head_dim_dec,
            dtype=dtype,
            device=device,
        )

        # ---- Cross-attention cache (static, length = encoder_seq_len) ----
        self.cross_attention_cache = WhisperCache(
            config=self.config,
            max_cache_len=encoder_seq_len,
        )

        self.cross_attention_cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_heads_dec,
            head_dim=head_dim_dec,
            dtype=dtype,
            device=device,
        )

        # Combine into an EncoderDecoderCache that uses static caches for both
        # self- and cross-attention.
        self.cache = EncoderDecoderCache(self.self_attention_cache, self.cross_attention_cache)

        # CRITICAL: Mark cross-attention cache as updated for all layers AFTER EncoderDecoderCache
        # is created, so Whisper will use the preloaded cache instead of recomputing K/V from encoder_hidden_states.
        # We must do this AFTER the EncoderDecoderCache.__init__() because that constructor
        # initializes is_updated based on cache sequence length (which is 0 initially).
        num_layers = len(self.cross_attention_cache.layers)
        for layer_idx in range(num_layers):
            self.cache.is_updated[layer_idx] = True

        # Register cache tensors as buffers on each decoder layer
        # (one buffer per layer for self- and cross-attn).
        for layer_idx, layer in enumerate(self.decoder.layers):
            self_layer = self.self_attention_cache.layers[layer_idx]
            cross_layer = self.cross_attention_cache.layers[layer_idx]

            layer.register_buffer(
                "self_attention_key_cache",
                self_layer.keys,
                persistent=False,
            )
            layer.register_buffer(
                "self_attention_value_cache",
                self_layer.values,
                persistent=False,
            )
            layer.register_buffer(
                "cross_attention_key_cache",
                cross_layer.keys,
                persistent=False,
            )
            layer.register_buffer(
                "cross_attention_value_cache",
                cross_layer.values,
                persistent=False,
            )

        # ---- Precomputed causal attention mask (for decoder self-attn) ----
        # Shape: (1, 1, T_max, T_max)
        min_val = torch.finfo(dtype).min
        base = torch.full(
            (max_static_cache_length, max_static_cache_length),
            fill_value=min_val,
            dtype=dtype,
            device=device,
        )
        base = torch.triu(base, diagonal=1)  # 0 on/below diag, -inf above
        causal_mask = base.view(1, 1, max_static_cache_length, max_static_cache_length)

        self.register_buffer("decoder_causal_mask", causal_mask, persistent=False)

    def _slice_causal_mask(
            self,
        batch_size: int,
        seq_len: int,
        cache_position_int: int,
    ) -> torch.Tensor:
        """
        Returns a Bx1xseq_lenxT_max causal mask slice from the precomputed buffer.

        We **do not** shorten the KV dimension; instead we:
        - keep KV length = max_cache_len (static cache length),
        - select rows corresponding to absolute positions
            [cache_position_int, ..., cache_position_int + seq_len - 1].

        This matches the static-cache behavior where K/V have length
        `max_cache_len` at every step.
        """
        T_max = self.max_cache_len

        # Compute row range [start, end) for this block
        start = cache_position_int
        end = start + seq_len
        torch._check(end <= T_max)

        # self.decoder_causal_mask: (1, 1, T_max, T_max)
        # We take rows [start:end] and all columns [:T_max]
        mask = self.decoder_causal_mask[:, :, start:end, :T_max]  # (1, 1, seq_len, T_max)

        if batch_size != 1:
            mask = mask.expand(batch_size, -1, -1, -1)

        return mask


    def _load_cross_kv_into_static_cache(
        self,
        cross_k_cache: torch.Tensor,  # (L, B, H, T_enc, D)
        cross_v_cache: torch.Tensor,  # (L, B, H, T_enc, D)
    ) -> None:
        """
        Copy the precomputed cross-attention K/V into the static
        cross_attention_cache for all layers, using narrow + copy_.
        """
        num_layers = len(self.cross_attention_cache.layers)

        torch._check(cross_k_cache.shape[0] == num_layers)
        torch._check(cross_v_cache.shape[0] == num_layers)

        _, B, H, T_enc, D = cross_k_cache.shape

        for layer_idx, layer_cache in enumerate(self.cross_attention_cache.layers):
            k_dst = layer_cache.keys   # (B, H, max_enc_len, D)
            v_dst = layer_cache.values

            k_src = cross_k_cache[layer_idx]  # (B, H, T_enc, D)
            v_src = cross_v_cache[layer_idx]

            torch._check(k_src.shape == (B, H, T_enc, D))
            torch._check(v_src.shape == (B, H, T_enc, D))

            # Overwrite beginning of static cross cache with the precomputed K/V
            k_dst.narrow(2, 0, T_enc).copy_(k_src)
            v_dst.narrow(2, 0, T_enc).copy_(v_src)

    def load_cross_kv_cache(
        self,
        cross_k_cache: torch.Tensor,  # (L, B, H, T_enc, D)
        cross_v_cache: torch.Tensor,  # (L, B, H, T_enc, D)
    ) -> None:
        """
        Load cross-attention K/V into buffers. Call this ONCE before generation starts.
        
        Args:
            cross_k_cache: (num_layers, batch, num_heads, encoder_seq_len, head_dim)
            cross_v_cache: (num_layers, batch, num_heads, encoder_seq_len, head_dim)
        """
        self._load_cross_kv_into_static_cache(cross_k_cache, cross_v_cache)

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,   # (B, T_dec)
        cache_position: torch.Tensor,          # 1D tensor containing start index
        encoder_hidden_states: torch.FloatTensor,  # (B, T_enc, H) - for API compatibility
    ) -> torch.FloatTensor:
        """
        Decoder forward pass. Cross K/V must be loaded via load_cross_kv_cache() first.
        
        Args:
            decoder_input_ids: (batch, seq_len) decoder input token IDs
            cache_position: 1D tensor with start index in self-attention cache
            encoder_hidden_states: encoder outputs (for API compatibility)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T_dec = decoder_input_ids.shape

        # Self-attention cache positioning
        torch._check(isinstance(cache_position, torch.Tensor))
        torch._check(cache_position.numel() == 1)
        cache_position_int = cache_position.item()
        torch._check_is_size(cache_position_int)

        attn_mask = self._slice_causal_mask(
            batch_size=B,
            seq_len=T_dec,
            cache_position_int=cache_position_int,
        )

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attn_mask,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = self.proj_out(outputs[0])
        return logits


# ---------------------------------------------------------------------
# Top-level helper: export Whisper encoder + decoder (+ cross-attn proj)
# ---------------------------------------------------------------------


def _make_example_encoder_input(
    model: WhisperForConditionalGeneration,
    batch_size: int,
) -> torch.Tensor:
    """
    Build a dummy encoder input of the right shape using the HF processor.
    """
    processor = AutoProcessor.from_pretrained(model.config._name_or_path)
    fe = processor.feature_extractor

    # Whisper feature shape: (B, feature_size=80, nb_max_frames)
    example = torch.zeros(
        (batch_size, fe.feature_size, fe.nb_max_frames),
        device=model.device,
        dtype=model.dtype,
    )
    return example


def export_whisper_encoder_decoder(
    model: WhisperForConditionalGeneration,
    *,
    batch_size: int = 1,
    max_decoder_seq_len: int = 448,
    model_dtype: torch.dtype = torch.float32,
    quantize: bool = False,
) -> Dict[str, ExportedProgram]:
    """
    Export a Whisper encoder and decoder as three torch.export ExportedPrograms.

    Returns:
        {
            "encoder": ExportedProgram,   # forward(input_features) -> encoder_hidden_states
            "cross_kv": ExportedProgram,  # forward(encoder_hidden_states) -> (k_cache, v_cache)
            "decoder": ExportedProgram,   # forward(decoder_input_ids, cross_k_cache, cross_v_cache, cache_position:int) -> logits
        }
    """
    from torchao.quantization.quant_api import quantize_, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup
   
   

    model.eval()
    device = model.device

    # ---------------- Create ALL wrappers FIRST (before any quantization) ----------------
    # This is important because they share the underlying decoder model
    
    # Encoder wrapper
    encoder_input = _make_example_encoder_input(model, batch_size=batch_size).to(model_dtype)
    encoder_wrapper = WhisperEncoderExportable(model.get_encoder()).to(device).to(model_dtype).eval()

    # Get example encoder output for cross-KV projection
    with torch.no_grad():
        example_encoder_hidden_states = encoder_wrapper(encoder_input)
    encoder_seq_len = example_encoder_hidden_states.shape[1]

    # Cross-KV projection wrapper
    cross_proj_wrapper = WhisperCrossAttentionProjections(
        decoder=model.get_decoder()
    ).to(device).to(model_dtype).eval()

    # Decoder wrapper (shares the same decoder as cross_proj_wrapper)
    start_id = getattr(model.config, "decoder_start_token_id", 0)
    decoder_input_ids = torch.full(
        (batch_size, 1),
        fill_value=start_id,
        dtype=torch.long,
        device=device,
    )
    cache_position = torch.tensor([0], dtype=torch.int64, device=device)

    decoder_wrapper = WhisperDecoderWithStaticCache(
        model=model,
        max_static_cache_length=max_decoder_seq_len,
        batch_size=batch_size,
        encoder_seq_len=encoder_seq_len,
    ).to(device).to(model_dtype).eval()

    # ---------------- Quantize wrappers (after all .to() calls) ----------------
    # IMPORTANT: cross_proj_wrapper and decoder_wrapper share the same underlying decoder
    # So we only quantize once through decoder_wrapper (which includes the full decoder + lm_head)
    if quantize:
        logger.info("Quantizing encoder and decoder wrappers...")
        quantize_(encoder_wrapper, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(64)))
        # Skip cross_proj_wrapper since it shares decoder with decoder_wrapper
        quantize_(decoder_wrapper, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(64)))

    # ---------------- Export encoder ----------------
    logger.info(f"Exporting Whisper encoder with input_features.shape={encoder_input.shape}")

    with torch.no_grad():
        encoder_ep: ExportedProgram = torch.export.export(
            encoder_wrapper,
            args=(encoder_input,),
            dynamic_shapes=None,
            strict=True,
        )
        encoder_ep = encoder_ep.run_decompositions({})

    # ---------------- Export cross-KV projections ----------------
    logger.info(
        "Exporting Whisper cross-attention projections with "
        f"encoder_hidden_states.shape={example_encoder_hidden_states.shape}, "
        f"encoder_seq_len={encoder_seq_len}"
    )

    with torch.no_grad():
        example_cross_k_cache, example_cross_v_cache = cross_proj_wrapper(example_encoder_hidden_states)

        cross_kv_ep: ExportedProgram = torch.export.export(
            cross_proj_wrapper,
            args=(example_encoder_hidden_states,),
            dynamic_shapes=None,
            strict=True,
        )
        cross_kv_ep = cross_kv_ep.run_decompositions({})

    # ---------------- Export decoder ----------------
    # Register custom HF attention implementation
    register_whisper_attention(model)

    logger.info(
        "Exporting Whisper decoder with "
        f"decoder_input_ids.shape={decoder_input_ids.shape}, "
        f"encoder_seq_len={encoder_seq_len}, "
        f"cross_k_tuple: {len(example_cross_k_cache)} tensors, first shape={example_cross_k_cache[0].shape}, "
        f"cache_position={cache_position}"
    )

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        decoder_ep: ExportedProgram = torch.export.export(
            decoder_wrapper,
            args=(decoder_input_ids, cache_position, example_encoder_hidden_states),
            dynamic_shapes=None,
            strict=True,
        )
        decoder_ep = decoder_ep.run_decompositions({})

    return {
        "encoder": encoder_ep,
        "cross_kv": cross_kv_ep,
        "decoder": decoder_ep,
    }



# ---------------------------------------------------------------------
# Tiny example usage (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_id = "openai/whisper-large-v3-turbo"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    eps = export_whisper_encoder_decoder(
        model,
        **DEFAULT_EXPORT_KWARGS
    )

    encoder_ep = eps["encoder"]
    cross_kv_ep = eps["cross_kv"]
    decoder_ep = eps["decoder"]

    print("Encoder graph:", encoder_ep)
    print("Cross-KV graph:", cross_kv_ep)
    print("Decoder graph:", decoder_ep)


    from program_builder import ProgramBuilder
    for k in eps:
        P = ProgramBuilder(eps[k])
        prog_json = P.build()
        with open(f"whisper_{k}_prog.json", "w") as f:
            f.write(prog_json)
        P.save_constant_data(f"whisper_{k}_consts.safetensors")
