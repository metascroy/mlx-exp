import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration

# ------------------------------
# 1) MODEL + PROCESSOR
# ------------------------------

model_id = "openai/whisper-large-v3-turbo"

processor = AutoProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.float32
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ------------------------------
# 2) LOAD LONG AUDIO SAMPLE
# ------------------------------

dataset = load_dataset(
    "distil-whisper/librispeech_long",
    "clean",
    split="validation",
)

sample = dataset[0]["audio"]
input_features = processor(
    sample["array"],
    return_tensors="pt",
    truncation=False,
    sampling_rate=sample["sampling_rate"],
).input_features

# Trim to 30 seconds
input_features_trimmed = input_features[:, :, :3000].contiguous()

# dtype + device
input_features_export = input_features_trimmed.to(device=device, dtype=model.dtype)

print("Input features:", input_features_export.shape)

# ------------------------------
# 2.5) SAVE ENCODER INPUT TO FILE
# ------------------------------

# Save encoder input for C++ to use
encoder_input_np = input_features_export.cpu().numpy()
encoder_input_np.tofile("whisper_encoder_input.bin")

# Save shape to a separate file
with open("whisper_encoder_input_shape.txt", "w") as f:
    shape = encoder_input_np.shape
    f.write(f"{shape[0]} {shape[1]} {shape[2]}\n")

print(f"Saved encoder input to whisper_encoder_input.bin (shape: {encoder_input_np.shape})")

# ------------------------------
# 3) IMPORT YOUR WRAPPERS
# ------------------------------

from export_whisper import (
    export_whisper_encoder_decoder,
    DEFAULT_EXPORT_KWARGS,
)


eps = export_whisper_encoder_decoder(model, **DEFAULT_EXPORT_KWARGS)

encoder_ep = eps["encoder"]
cross_kv_ep = eps["cross_kv"]
decoder_ep = eps["decoder"]


# ------------------------------
# 4) RUN ENCODER
# ------------------------------

encoder_wrapper = encoder_ep.module()

print("input_features_export", input_features_export)
with torch.no_grad():
    encoder_hidden_states = encoder_wrapper(input_features_export)

print("encoder_hidden_states:", encoder_hidden_states.shape)  # (1, T_enc, H)
print("encoder_hidden_states", encoder_hidden_states)

# ------------------------------
# 5) PRECOMPUTE CROSS-KV
# ------------------------------

cross_proj_wrapper = cross_kv_ep.module()

with torch.no_grad():
    cross_k_tuple, cross_v_tuple = cross_proj_wrapper(encoder_hidden_states)

print(f"cross_k_tuple: {len(cross_k_tuple)} tensors, each shape {cross_k_tuple[0].shape}")
print(f"cross_v_tuple: {len(cross_v_tuple)} tensors, each shape {cross_v_tuple[0].shape}")

# ------------------------------
# 6) SETUP DECODER (STATIC CACHE)
# ------------------------------

decoder = decoder_ep.module()

# Manually load cross-attention K/V into the exported module's buffers
# The buffers are accessible as decoder.layers[i].cross_attention_key_cache/value_cache
print("Loading cross-attention K/V into decoder buffers...")
num_layers = len(cross_k_tuple)
for layer_idx in range(num_layers):
    layer_module = decoder.get_submodule(f"decoder.layers.{layer_idx}")
    # Get the cross K/V for this layer from the tuples
    cross_k = cross_k_tuple[layer_idx]
    cross_v = cross_v_tuple[layer_idx]
    T_enc = cross_k.shape[2]  # encoder sequence length
    # Direct copy without indexing - just need to narrow to actual encoder length
    layer_module.cross_attention_key_cache.narrow(2, 0, T_enc).copy_(cross_k)
    layer_module.cross_attention_value_cache.narrow(2, 0, T_enc).copy_(cross_v)
print("Cross-attention K/V loaded successfully!")

# ------------------------------
# 7) TOKEN-BY-TOKEN DECODE LOOP
# ------------------------------

# 7.1 Get forced decoder ids - use processor.get_decoder_prompt_ids() for complete config
# model.generation_config.forced_decoder_ids is incomplete ([[1, None], [2, 50360]])
# processor.get_decoder_prompt_ids() gives the full config: [(1, 50259), (2, 50360), (3, 50364)]
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en",
    task="transcribe",
)

print("Raw forced_decoder_ids:", forced_decoder_ids)

# Whisper: decoder_start_token_id is <|startoftranscript|>
sot_id = model.config.decoder_start_token_id

# For Whisper, the initial prompt is just the start-of-transcript token
# The forced_decoder_ids specify tokens that should be forced at specific positions DURING generation
# We should NOT include them in the initial prompt
prompt_ids = [sot_id]

print("Prompt ids:", prompt_ids)
print("Prompt tokens:", processor.tokenizer.convert_ids_to_tokens(prompt_ids))
print("Forced decoder ids (will be applied during generation):", forced_decoder_ids)

# ------------------------------
# 7.2) SAVE PROMPT TOKENS TO FILE
# ------------------------------

# Save prompt tokens for C++ to use
with open("whisper_prompt_ids.txt", "w") as f:
    f.write(" ".join(map(str, prompt_ids)) + "\n")

# Save forced tokens dict for reference
with open("whisper_forced_tokens.txt", "w") as f:
    for pos, tok_id in forced_decoder_ids:
        if tok_id is not None:
            f.write(f"{pos} {tok_id}\n")

print(f"Saved prompt tokens to whisper_prompt_ids.txt: {prompt_ids}")
print(f"Saved forced tokens to whisper_forced_tokens.txt")

# ------------------------------
# 7.3) DECODER GENERATION
# ------------------------------

decoder_input_ids = torch.tensor(
    [prompt_ids],  # shape (1, T_prompt) = (1, 1)
    device=device,
    dtype=torch.long,
)

generated: list[int] = prompt_ids.copy()
cache_position = torch.tensor([0], dtype=torch.int64, device=device)

max_new_tokens = 256

# Apply forced decoder IDs manually during generation
forced_tokens_dict = {}
if forced_decoder_ids is not None:
    for item in forced_decoder_ids:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pos, tok_id = item
            if tok_id is not None:
                forced_tokens_dict[pos] = int(tok_id)

print("Forced tokens dict:", forced_tokens_dict)

with torch.no_grad():
    # First call: prefill with just the SOT token
    logits = decoder(
        decoder_input_ids,
        cache_position,
        encoder_hidden_states,
    )
    T_prompt = decoder_input_ids.shape[1]
    cache_position = cache_position + T_prompt
    
    print(f"\nAfter prefill: T_prompt={T_prompt}, cache_position={cache_position.item()}")
    print(f"Logits shape: {logits.shape}")  # Should be (1, 1, vocab_size)

    # Incremental generation
    for step in range(max_new_tokens):
        # Check if this position has a forced token
        current_position = cache_position.item()
        if current_position in forced_tokens_dict:
            next_token_id = forced_tokens_dict[current_position]
            if step < 5:
                print(f"[step {step}] cache_position={current_position}, FORCED token: {next_token_id} ->", 
                      processor.tokenizer.convert_ids_to_tokens([next_token_id]))
        else:
            # Sample from the last position of the logits
            next_token_id = torch.argmax(logits[:, -1], dim=-1).item()
            if step < 5:
                print(f"[step {step}] cache_position={current_position}, SAMPLED token: {next_token_id} ->",
                      processor.tokenizer.convert_ids_to_tokens([next_token_id]))

        generated.append(next_token_id)

        if next_token_id == model.config.eos_token_id:
            print("Encountered EOS token; will stop after this step.")
            break

        decoder_input_ids = torch.tensor(
            [[next_token_id]],
            device=device,
            dtype=torch.long,
        )

        logits = decoder(
            decoder_input_ids,
            cache_position,
            encoder_hidden_states,
        )
        cache_position = cache_position + 1

print("Generated ids:", generated)
print("Generated tokens:", processor.tokenizer.convert_ids_to_tokens(generated))

text_no_strip = processor.tokenizer.decode(
    generated,
    skip_special_tokens=False,
)
text_strip = processor.tokenizer.decode(
    generated,
    skip_special_tokens=True,
)

print("\n\n ===== RAW (no strip) ===== \n")
print(repr(text_no_strip))

print("\n\n ===== FINAL TRANSCRIPT (skip_special_tokens=True) ===== \n")
print(text_strip)
