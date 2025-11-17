// whisper_main.cpp — run exported Whisper encoder + cross-kv projector + decoder JSON programs
#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"
#include "program_json_loader.hpp"

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/memory.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

using namespace executorch::mlx;
using namespace ::mlx::core;

// ---------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------
static std::string env_or(const char* key, const char* defval) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : std::string(defval);
}
static int env_or_int(const char* key, int defval) {
  const char* v = std::getenv(key);
  return v ? std::max(0, std::atoi(v)) : defval;
}

// ---------------------------------------------------------------------
// logits: [B, T, V] -> argmax(last) -> [B,1] int32
// ---------------------------------------------------------------------
static array sample_next_token(const array& logits) {
  auto s = logits.shape();
  if (s.size() != 3) throw std::runtime_error("logits must be [B,T,V]");
  int B = s[0], T = s[1], V = s[2];
  array last = (T > 1)
      ? slice(logits, Shape{0, T - 1, 0}, Shape{B, T, V})
      : logits;
  array idx = argmax(last, /*axis=*/2);
  if (idx.dtype() != int32) idx = astype(idx, int32);
  return idx;  // [B,1]
}

// ---------------------------------------------------------------------
// Load a JSON program from file
// ---------------------------------------------------------------------
static Program load_program(const std::string& json_path) {
  std::ifstream jf(json_path);
  if (!jf) throw std::runtime_error("cannot open prog.json: " + json_path);
  nlohmann::json j;
  jf >> j;
  return program_from_json(j);
}

int main() {
  try {
    // -----------------------------------------------------------------
    // Paths for encoder, cross-kv projector, and decoder
    // -----------------------------------------------------------------
    const std::string encoder_prog_json   = env_or("ENCODER_PROG_JSON",   "/Users/scroy/repos/mlx-exp/whisper_encoder_prog.json");
    const std::string encoder_consts_path = env_or("ENCODER_CONSTS_ST",   "/Users/scroy/repos/mlx-exp/whisper_encoder_consts.safetensors");

    const std::string cross_kv_prog_json   = env_or("CROSS_KV_PROG_JSON",   "/Users/scroy/repos/mlx-exp/whisper_cross_kv_prog.json");
    const std::string cross_kv_consts_path = env_or("CROSS_KV_CONSTS_ST",   "/Users/scroy/repos/mlx-exp/whisper_cross_kv_consts.safetensors");

    const std::string decoder_prog_json   = env_or("DECODER_PROG_JSON",   "/Users/scroy/repos/mlx-exp/whisper_decoder_prog.json");
    const std::string decoder_consts_path = env_or("DECODER_CONSTS_ST",   "/Users/scroy/repos/mlx-exp/whisper_decoder_consts.safetensors");

    const std::string encoder_input_bin   = env_or("ENCODER_INPUT_BIN",   "/Users/scroy/repos/mlx-exp/whisper_encoder_input.bin");
    const std::string encoder_input_shape = env_or("ENCODER_INPUT_SHAPE", "/Users/scroy/repos/mlx-exp/whisper_encoder_input_shape.txt");
    const std::string forced_tokens_file  = env_or("FORCED_TOKENS_FILE",  "/Users/scroy/repos/mlx-exp/whisper_forced_tokens.txt");
    const std::string prompt_ids_file     = env_or("PROMPT_IDS_FILE",     "/Users/scroy/repos/mlx-exp/whisper_prompt_ids.txt");

    const int max_new_tokens      = env_or_int("MAX_NEW_TOKENS", 128);
    const int print_batch         = std::max(1, env_or_int("PRINT_BATCH", 1));
    const std::string output_ids  = env_or("OUTPUT_IDS",  "/Users/scroy/repos/mlx-exp/output_ids.txt");

    set_wired_limit(4L * (1 << 30));  // 4GB

    // -----------------------------------------------------------------
    // 1. Load encoder program
    // -----------------------------------------------------------------
    std::cout << "\n========== LOADING ENCODER ==========\n";
    Program P_encoder = load_program(encoder_prog_json);

    ConstantData store_encoder;
    bind_constants_from_safetensors(encoder_consts_path, P_encoder, store_encoder);

    ExecutionState S_encoder;
    S_encoder.bind(P_encoder);
    init_execution_state_from_meta(P_encoder, S_encoder);

    std::cout << "[encoder] constant tensors: " << P_encoder.num_constant_tensors << "\n";
    std::cout << "[encoder] non-constant tensors: " << P_encoder.num_non_constant_tensors << "\n";
    std::cout << "[encoder] inputs: " << P_encoder.num_inputs()
              << ", outputs: " << P_encoder.num_outputs() << "\n";

    // Eval constants
    for (size_t i = 0; i < P_encoder.constants->tensors.size(); ++i) {
      eval(P_encoder.constants->tensors.at(i));
    }

    // -----------------------------------------------------------------
    // 2. Load cross-kv projector program
    // -----------------------------------------------------------------
    std::cout << "\n========== LOADING CROSS-KV PROJECTOR ==========\n";
    Program P_cross_kv = load_program(cross_kv_prog_json);

    ConstantData store_cross_kv;
    bind_constants_from_safetensors(cross_kv_consts_path, P_cross_kv, store_cross_kv);

    ExecutionState S_cross_kv;
    S_cross_kv.bind(P_cross_kv);
    init_execution_state_from_meta(P_cross_kv, S_cross_kv);

    std::cout << "[cross_kv] constant tensors: " << P_cross_kv.num_constant_tensors << "\n";
    std::cout << "[cross_kv] non-constant tensors: " << P_cross_kv.num_non_constant_tensors << "\n";
    std::cout << "[cross_kv] inputs: " << P_cross_kv.num_inputs()
              << ", outputs: " << P_cross_kv.num_outputs() << "\n";

    // Eval constants
    for (size_t i = 0; i < P_cross_kv.constants->tensors.size(); ++i) {
      eval(P_cross_kv.constants->tensors.at(i));
    }

    // -----------------------------------------------------------------
    // 3. Load decoder program
    // -----------------------------------------------------------------
    std::cout << "\n========== LOADING DECODER ==========\n";
    Program P_decoder = load_program(decoder_prog_json);

    ConstantData store_decoder;
    bind_constants_from_safetensors(decoder_consts_path, P_decoder, store_decoder);

    ExecutionState S_decoder;
    S_decoder.bind(P_decoder);
    init_execution_state_from_meta(P_decoder, S_decoder);

    std::cout << "[decoder] constant tensors: " << P_decoder.num_constant_tensors << "\n";
    std::cout << "[decoder] non-constant tensors: " << P_decoder.num_non_constant_tensors << "\n";
    std::cout << "[decoder] inputs: " << P_decoder.num_inputs()
              << ", outputs: " << P_decoder.num_outputs() << "\n";

    // Eval decoder constants
    for (size_t i = 0; i < P_decoder.constants->tensors.size(); ++i) {
      eval(P_decoder.constants->tensors.at(i));
    }

    Interpreter I;

    // -----------------------------------------------------------------
    // 4. Run encoder
    // -----------------------------------------------------------------
    std::cout << "\n========== RUNNING ENCODER ==========\n";

    if (P_encoder.num_inputs() != 1) {
      throw std::runtime_error("encoder program must have 1 input");
    }
    if (P_encoder.num_outputs() != 1) {
      throw std::runtime_error("encoder program must have 1 output");
    }

    auto encoder_input_tid = std::get<Tid>(P_encoder.input_map[0]);
    auto encoder_out_tid = std::get<Tid>(P_encoder.output_map[0]);

    // Load encoder input from file (shape: B, feature_size, nb_max_frames)
    std::ifstream shape_in(encoder_input_shape);
    if (!shape_in) {
      throw std::runtime_error("Could not open encoder input shape file: " + encoder_input_shape);
    }
    int B, F, T;
    shape_in >> B >> F >> T;

    std::ifstream bin_in(encoder_input_bin, std::ios::binary);
    if (!bin_in) {
      throw std::runtime_error("Could not open encoder input binary file: " + encoder_input_bin);
    }
    std::vector<float> buf(B * F * T);
    bin_in.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));

    array tmp(buf.data(), Shape{B, F, T}, float32);
    array encoder_in = tmp;  // Keep as float32
    eval(encoder_in);
    S_encoder.tensor_ref(encoder_input_tid) = encoder_in;

    auto t0_enc = std::chrono::steady_clock::now();
    I.run(P_encoder, S_encoder);
    const auto& encoder_hidden_states = S_encoder.const_tensor_ref(encoder_out_tid);
    eval(encoder_hidden_states);
    auto t1_enc = std::chrono::steady_clock::now();

    double encoder_ms = std::chrono::duration<double, std::milli>(t1_enc - t0_enc).count();
    std::cout << "Encoder time: " << encoder_ms << " ms\n";

    // Save encoder output to file for comparison
    {
      std::ofstream outfile("cpp_encoder_output.bin", std::ios::binary);
      if (outfile) {
        // Convert to float32 if needed and save
        array out_f32 = (encoder_hidden_states.dtype() == float32) ? encoder_hidden_states : astype(encoder_hidden_states, float32);
        eval(out_f32);
        const float* data_ptr = out_f32.data<float>();
        size_t total_elements = out_f32.size();
        outfile.write(reinterpret_cast<const char*>(data_ptr), total_elements * sizeof(float));
      }
    }

    // -----------------------------------------------------------------
    // 5. Run cross-kv projector to precompute cross-attention K/V
    // -----------------------------------------------------------------
    std::cout << "\n========== RUNNING CROSS-KV PROJECTOR ==========\n";

    if (P_cross_kv.num_inputs() != 1) {
      throw std::runtime_error("cross_kv program must have 1 input");
    }

    auto cross_kv_input_tid = std::get<Tid>(P_cross_kv.input_map[0]);
    S_cross_kv.tensor_ref(cross_kv_input_tid) = encoder_hidden_states;

    auto t0_kv = std::chrono::steady_clock::now();
    I.run(P_cross_kv, S_cross_kv);

    // Outputs are cross_k and cross_v for each layer
    // Assuming outputs are ordered: cross_k_0, cross_v_0, cross_k_1, cross_v_1, ...
    std::cout << "Cross-KV outputs: " << P_cross_kv.num_outputs() << "\n";

    // Collect cross-attention K/V into vectors
    std::vector<array> cross_k_list, cross_v_list;
    int num_layers = P_cross_kv.num_outputs() / 2;

    // The cross_kv program outputs all K first, then all V
    // Outputs 0 to (num_layers-1): cross_k for layers 0, 1, 2, ...
    // Outputs num_layers to (2*num_layers-1): cross_v for layers 0, 1, 2, ...
    for (int layer = 0; layer < num_layers; ++layer) {
      auto cross_k_tid = std::get<Tid>(P_cross_kv.output_map[layer]);
      auto cross_v_tid = std::get<Tid>(P_cross_kv.output_map[num_layers + layer]);

      const auto& cross_k = S_cross_kv.const_tensor_ref(cross_k_tid);
      const auto& cross_v = S_cross_kv.const_tensor_ref(cross_v_tid);

      eval(cross_k);
      eval(cross_v);

      cross_k_list.push_back(cross_k);
      cross_v_list.push_back(cross_v);
    }

    auto t1_kv = std::chrono::steady_clock::now();
    double kv_ms = std::chrono::duration<double, std::milli>(t1_kv - t0_kv).count();
    std::cout << "Cross-KV time: " << kv_ms << " ms\n";

    // -----------------------------------------------------------------
    // 6. Initialize decoder mutable buffers
    // -----------------------------------------------------------------
    std::cout << "\n========== INITIALIZING DECODER BUFFERS ==========\n";

    std::cout << "Decoder mutable buffers: " << P_decoder.mutable_buffer_map.size() << "\n";

    // Initialize all mutable buffers from name_to_slot
    // We need to map buffer names to their slots and initialize them
    std::unordered_map<std::string, Tid> buffer_name_to_tid;

    for (const auto& [name, slot_var] : P_decoder.nameToSlot) {
      if (!std::holds_alternative<Tid>(slot_var)) continue;
      Tid tid = std::get<Tid>(slot_var);

      // Check if this tid is in mutable_buffer_map
      bool is_mutable_buffer = false;
      for (const auto& buf_slot_var : P_decoder.mutable_buffer_map) {
        if (!std::holds_alternative<Tid>(buf_slot_var)) continue;
        Tid buf_tid = std::get<Tid>(buf_slot_var);
        if (buf_tid.idx == tid.idx) {
          is_mutable_buffer = true;
          break;
        }
      }

      if (is_mutable_buffer) {
        buffer_name_to_tid[name] = tid;
      }
    }

    std::cout << "Found " << buffer_name_to_tid.size() << " named mutable buffers\n";

    // Initialize cross-attention K/V buffers
    int cross_kv_loaded = 0;
    for (int layer = 0; layer < num_layers; ++layer) {
      // Look for cross_attention_key_cache and cross_attention_value_cache for this layer
      std::string k_name = "decoder.layers." + std::to_string(layer) + ".cross_attention_key_cache";
      std::string v_name = "decoder.layers." + std::to_string(layer) + ".cross_attention_value_cache";

      auto k_it = buffer_name_to_tid.find(k_name);
      auto v_it = buffer_name_to_tid.find(v_name);

      if (k_it != buffer_name_to_tid.end() && v_it != buffer_name_to_tid.end()) {
        S_decoder.tensor_ref(k_it->second) = cross_k_list[layer];
        S_decoder.tensor_ref(v_it->second) = cross_v_list[layer];
        cross_kv_loaded += 2;
      }
    }
    std::cout << "Loaded " << cross_kv_loaded << " cross-attention K/V buffers (" << (cross_kv_loaded/2) << " layers)\n";

    // Helper lambda to get Tid by name
    auto get_tid_by_name = [&](const std::string& name) -> std::optional<Tid> {
      auto it = buffer_name_to_tid.find(name);
      if (it == buffer_name_to_tid.end()) {
        return std::nullopt;
      }
      return it->second;
    };

    // Helper lambda to initialize causal mask buffer
    auto initialize_causal_mask = [&](Tid tid) -> bool {
      if (tid.idx >= P_decoder.tensor_meta.size() || !P_decoder.tensor_meta[tid.idx].has_value()) {
        return false;
      }

      const auto& meta = *P_decoder.tensor_meta[tid.idx];
      std::vector<int64_t> shape64;
      for (int d : meta.shape) shape64.push_back(d);

      // Causal mask should be 4D with square last two dimensions
      if (shape64.size() != 4 || shape64[2] != shape64[3]) {
        return false;
      }

      auto dtype = to_mlx(meta.dtype);
      int seq_len = shape64[2];

      // Create causal mask using MLX operations (similar to mlx-lm create_causal_mask)
      // rinds = arange(seq_len)  -> [0, 1, 2, ..., seq_len-1]
      // linds = arange(seq_len)  -> [0, 1, 2, ..., seq_len-1]
      // linds[:, None] -> column vector
      // rinds[None, :] -> row vector
      // mask = linds >= rinds creates lower triangular (including diagonal)

      array rinds = arange(seq_len);  // [seq_len]
      array linds = arange(seq_len);  // [seq_len]

      // Reshape for broadcasting: linds as column [seq_len, 1], rinds as row [1, seq_len]
      linds = reshape(linds, {seq_len, 1});
      rinds = reshape(rinds, {1, seq_len});

      // Create boolean mask: linds >= rinds
      // This gives True for positions where we can attend (lower triangular + diagonal)
      array mask = greater_equal(linds, rinds);  // [seq_len, seq_len], bool

      // For attention, we need: 0 for positions we CAN attend to, -inf for positions we CANNOT
      // So we need to invert: where mask is True (can attend) -> 0, where False (cannot) -> -inf
      // Create directly in the target dtype (bf16 for efficiency)
      array zeros_arr = zeros({seq_len, seq_len}, dtype);
      array neginf_arr = full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(), dtype);
      array mask_float = where(mask, zeros_arr, neginf_arr);  // [seq_len, seq_len], target dtype

      // Reshape to [1, 1, seq_len, seq_len]
      array buf = reshape(mask_float, Shape{1, 1, seq_len, seq_len});

      eval(buf);
      S_decoder.tensor_ref(tid) = buf;
      std::cout << "✅ Initialized causal mask shape [1,1," << seq_len << "," << seq_len
                << "] with -inf for upper triangular (dtype: ";
      switch (dtype) {
        case float32: std::cout << "float32"; break;
        case bfloat16: std::cout << "bfloat16"; break;
        case float16: std::cout << "float16"; break;
        default: std::cout << "other"; break;
      }
      std::cout << ")\n";
      return true;
    };

    // Initialize causal mask by name
    int other_buffers_initialized = 0;
    auto causal_mask_tid = get_tid_by_name("decoder_causal_mask");
    if (causal_mask_tid.has_value()) {
      if (initialize_causal_mask(*causal_mask_tid)) {
        other_buffers_initialized++;
      } else {
        std::cerr << "WARNING: Failed to initialize causal mask\n";
      }
    } else {
      std::cerr << "WARNING: decoder_causal_mask not found in program\n";
    }

    // Initialize remaining uninitialized mutable buffers with zeros
    for (const auto& buf_slot_var : P_decoder.mutable_buffer_map) {
      if (!std::holds_alternative<Tid>(buf_slot_var)) continue;
      Tid tid = std::get<Tid>(buf_slot_var);

      // Check if already initialized
      uint32_t slot_idx = tid.idx - P_decoder.num_constant_tensors;
      if (slot_idx < S_decoder.tensors.size() && S_decoder.tensors[slot_idx].has_value()) {
        continue;  // Already initialized
      }

      // Initialize from metadata with zeros
      if (tid.idx < P_decoder.tensor_meta.size() && P_decoder.tensor_meta[tid.idx].has_value()) {
        const auto& meta = *P_decoder.tensor_meta[tid.idx];
        std::vector<int64_t> shape64;
        for (int d : meta.shape) shape64.push_back(d);

        auto dtype = to_mlx(meta.dtype);
        array buf = zeros(Shape(shape64.begin(), shape64.end()), dtype);
        eval(buf);
        S_decoder.tensor_ref(tid) = buf;
        other_buffers_initialized++;
      }
    }

    std::cout << "Initialized " << other_buffers_initialized << " other mutable buffers\n";

    // -----------------------------------------------------------------
    // 7. Decoder loop: token-by-token generation with static KV cache
    // -----------------------------------------------------------------
    std::cout << "\n========== RUNNING DECODER LOOP ==========\n";

    // Check decoder inputs - could be 2 or 3 inputs depending on export
    // 2 inputs: decoder_input_ids, cache_position (encoder_hidden_states passed via buffers)
    // 3 inputs: decoder_input_ids, encoder_hidden_states, cache_position
    if (P_decoder.num_inputs() < 2 || P_decoder.num_inputs() > 3) {
      throw std::runtime_error("decoder program must have 2-3 inputs");
    }
    if (P_decoder.num_outputs() != 1) {
      throw std::runtime_error("decoder program must have 1 output (logits)");
    }

    Tid decoder_ids_tid;
    std::optional<Vid<int32_t>> decoder_cache_pos_vid;
    std::optional<Tid> decoder_cache_pos_tid;  // cache_position as tensor
    std::optional<Tid> decoder_enc_tid;

    // Parse input map based on number of inputs and types
    try {
      if (P_decoder.num_inputs() == 2) {
        // 2 inputs: decoder_input_ids, cache_position
        decoder_ids_tid = std::get<Tid>(P_decoder.input_map[0]);

        // cache_position could be Vid<int32_t> or Tid
        if (std::holds_alternative<Vid<int32_t>>(P_decoder.input_map[1])) {
          decoder_cache_pos_vid = std::get<Vid<int32_t>>(P_decoder.input_map[1]);
          std::cout << "Decoder has 2 inputs (ids, cache_pos as Vid)\n";
        } else if (std::holds_alternative<Tid>(P_decoder.input_map[1])) {
          decoder_cache_pos_tid = std::get<Tid>(P_decoder.input_map[1]);
          std::cout << "Decoder has 2 inputs (ids, cache_pos as Tid)\n";
        }
      } else {
        // 3 inputs: decoder_input_ids, encoder_hidden_states, cache_position
        decoder_ids_tid = std::get<Tid>(P_decoder.input_map[0]);
        decoder_enc_tid = std::get<Tid>(P_decoder.input_map[1]);

        // cache_position could be Vid<int32_t> or Tid
        if (std::holds_alternative<Vid<int32_t>>(P_decoder.input_map[2])) {
          decoder_cache_pos_vid = std::get<Vid<int32_t>>(P_decoder.input_map[2]);
          std::cout << "Decoder has 3 inputs (ids, enc_hidden, cache_pos as Vid)\n";
        } else if (std::holds_alternative<Tid>(P_decoder.input_map[2])) {
          decoder_cache_pos_tid = std::get<Tid>(P_decoder.input_map[2]);
          std::cout << "Decoder has 3 inputs (ids, enc_hidden, cache_pos as Tid)\n";
        }
      }
    } catch (const std::bad_variant_access& e) {
      std::cerr << "ERROR: bad_variant_access when parsing decoder inputs\n";
      std::cerr << "This means the input types don't match what we expected\n";
      throw;
    }

    auto decoder_logits_tid = std::get<Tid>(P_decoder.output_map[0]);

    // Helper to set cache_position (either as Vid or Tid)
    auto set_cache_position = [&](int32_t v) {
      if (decoder_cache_pos_vid.has_value()) {
        if (decoder_cache_pos_vid->idx >= S_decoder.values.size()) {
          throw std::out_of_range("set_cache_position: id out of range");
        }
        S_decoder.values[decoder_cache_pos_vid->idx] = Value{ v };
      } else if (decoder_cache_pos_tid.has_value()) {
        // Set as tensor (scalar int32)
        S_decoder.tensor_ref(*decoder_cache_pos_tid) = array(v, int32);
      }
    };

    // Bind encoder_hidden_states if needed as input
    if (decoder_enc_tid.has_value()) {
      S_decoder.tensor_ref(*decoder_enc_tid) = encoder_hidden_states;
    }

    // Load forced decoder IDs from file (generated by Python run_whisper.py)
    // The file contains position-token pairs: "1 50259\n2 50360\n3 50364\n"
    std::unordered_map<int, int> forced_tokens_dict;
    {
      std::ifstream forced_file(forced_tokens_file);
      if (!forced_file) {
        throw std::runtime_error("Could not load forced tokens file: " + forced_tokens_file);
      }
      int pos, tok_id;
      while (forced_file >> pos >> tok_id) {
        forced_tokens_dict[pos] = tok_id;
      }
      std::cout << "Loaded " << forced_tokens_dict.size() << " forced tokens from file\n";
    }

    // Load start-of-transcript token from prompt file
    int sot_token;
    {
      std::ifstream prompt_file(prompt_ids_file);
      if (!prompt_file) {
        throw std::runtime_error("Could not load prompt IDs file: " + prompt_ids_file);
      }
      if (!(prompt_file >> sot_token)) {
        throw std::runtime_error("Failed to read SOT token from: " + prompt_ids_file);
      }
      std::cout << "Loaded SOT token: " << sot_token << "\n";
    }

    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    array cur_ids = full(Shape{1, 1}, sot_token, int32);
    generated.push_back(sot_token);

    auto t0_dec = std::chrono::steady_clock::now();

    // First decoder call (prefill with SOT token at position 0)
    std::cout << "  [prefill] cache_position=0, input token: " << sot_token << "\n";
    S_decoder.tensor_ref(decoder_ids_tid) = cur_ids;
    set_cache_position(0);
    I.run(P_decoder, S_decoder);
    const auto& prefill_logits = S_decoder.const_tensor_ref(decoder_logits_tid);
    eval(prefill_logits);

    // Now generate tokens one by one
    for (int step = 0; step < max_new_tokens; ++step) {
      int current_position = step + 1;  // Position after prefill

      int next_token_id;

      // Check if this position has a forced token
      auto forced_it = forced_tokens_dict.find(current_position);
      if (forced_it != forced_tokens_dict.end()) {
        next_token_id = forced_it->second;
        if (step < 5) {
          std::cout << "  [step " << step << "] cache_position=" << current_position
                    << ", FORCED token: " << next_token_id << "\n";
        }
      } else {
        // Sample from the logits (argmax)
        const auto& logits = S_decoder.const_tensor_ref(decoder_logits_tid);
        array next_ids = sample_next_token(logits);
        eval(next_ids);
        next_token_id = next_ids.item<int>();
      }

      generated.push_back(next_token_id);

      // Check for EOS token (Whisper uses 50257)
      if (next_token_id == 50257) {
        std::cout << "  Encountered EOS token, stopping.\n";
        break;
      }

      // Prepare next input
      cur_ids = full(Shape{1, 1}, next_token_id, int32);
      S_decoder.tensor_ref(decoder_ids_tid) = cur_ids;
      set_cache_position(current_position);

      // Run decoder for next step
      I.run(P_decoder, S_decoder);

      if ((step + 1) % print_batch == 0) {
        std::cout << "  generated " << (step + 1) << " tokens; last id=" << next_token_id << "\n";
      }
    }
    auto t1_dec = std::chrono::steady_clock::now();

    double decoder_ms = std::chrono::duration<double, std::milli>(t1_dec - t0_dec).count();
    double tok_per_s  = (decoder_ms > 0.0)
                      ? (generated.size() / (decoder_ms / 1000.0))
                      : 0.0;

    std::cout << "\n========== RESULTS ==========\n";
    std::cout << "Encoder time: " << encoder_ms << " ms\n";
    std::cout << "Cross-KV time: " << kv_ms << " ms\n";
    std::cout << "Decoder time: " << decoder_ms << " ms for "
              << generated.size() << " tokens ("
              << tok_per_s << " tok/s)\n";
    std::cout << "Total time: " << (encoder_ms + kv_ms + decoder_ms) << " ms\n";

    std::cout << "\nGenerated " << generated.size() << " tokens: ";
    for (size_t i = 0; i < std::min(generated.size(), size_t(20)); ++i) {
      if (i) std::cout << ' ';
      std::cout << generated[i];
    }
    if (generated.size() > 20) std::cout << " ...";
    std::cout << "\n";

    // Write output ids if requested
    if (!output_ids.empty()) {
      std::ofstream ofs(output_ids);
      if (!ofs) {
        std::cerr << "warning: could not open OUTPUT_IDS file: " << output_ids << "\n";
      } else {
        for (size_t i = 0; i < generated.size(); ++i) {
          if (i) ofs << ' ';
          ofs << generated[i];
        }
        ofs << '\n';
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
