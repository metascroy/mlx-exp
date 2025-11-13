// whisper_main.cpp — run exported Whisper encoder + text decoder JSON programs
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
// Token file helper (not currently used, but kept for parity)
// ---------------------------------------------------------------------
static std::vector<int> read_token_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open tokens file: " + path);
  std::vector<int> ids; ids.reserve(8192);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int x;
    while (iss >> x) ids.push_back(x);
  }
  if (ids.empty()) throw std::runtime_error("no token ids in: " + path);
  return ids;
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

int main() {
  try {
    const std::string encoder_prog_json   = env_or("ENCODER_PROG_JSON",   "/Users/scroy/repos/mlx-exp/whisper_encoder_prog.json");
    const std::string encoder_consts_path = env_or("ENCODER_CONSTS_ST",   "/Users/scroy/repos/mlx-exp/whisper_encoder_consts.safetensors");

    const std::string decoder_prog_json   = env_or("DECODER_PROG_JSON",   "/Users/scroy/repos/mlx-exp/whisper_text_decoder_prog.json");
    const std::string decoder_consts_path = env_or("DECODER_CONSTS_ST",   "/Users/scroy/repos/mlx-exp/whisper_text_decoder_consts.safetensors");

    const int max_new_tokens      = env_or_int("MAX_NEW_TOKENS", 128);
    const int print_batch         = std::max(1, env_or_int("PRINT_BATCH", 1));
    const int bos_token_id        = env_or_int("BOS_TOKEN_ID", 1);
    const std::string output_ids  = env_or("OUTPUT_IDS",  "/Users/scroy/repos/mlx-exp/output_ids.txt");

    set_wired_limit(4L * (1 << 30));  // 4GB

    // -----------------------------------------------------------------
    // Load encoder program
    // -----------------------------------------------------------------
    std::ifstream jf_enc(encoder_prog_json);
    if (!jf_enc) throw std::runtime_error("cannot open prog.json: " + encoder_prog_json);
    nlohmann::json j_enc;
    jf_enc >> j_enc;
    Program P_encoder = program_from_json(j_enc);

    // load encoder constants
    ConstantData store_encoder;
    bind_constants_from_safetensors(encoder_consts_path, P_encoder, store_encoder);

    // Execution state
    ExecutionState S_encoder;
    S_encoder.bind(P_encoder);
    init_execution_state_from_meta(P_encoder, S_encoder);

    std::cout << "[encoder] constant tensors: " << P_encoder.num_constant_tensors << "\n";
    std::cout << "[encoder] non-constant tensors: " << P_encoder.num_non_constant_tensors << "\n";
    std::cout << "[encoder] non-constant values: " << P_encoder.num_non_constant_values << "\n";
    std::cout << "[encoder] inputs: " << P_encoder.num_inputs()
              << ", outputs: " << P_encoder.num_outputs() << "\n";

    // Eval constants before timing prefill
    for (size_t i = 0; i < P_encoder.constants->tensors.size(); ++i) {
      eval(P_encoder.constants->tensors.at(i));
    }

    Interpreter I;

    // Create random encoder input
    if (P_encoder.num_inputs() != 1) {
      throw std::runtime_error("encoder program must have 1 input");
    }
    auto encoder_input_tid = std::get<Tid>(P_encoder.input_map[0]);

    // Shape must match your exported encoder (B, feature_size, nb_max_frames)
    // S_encoder.tensor_ref(encoder_input_tid) = full(Shape{1, 128, 3000}, 1.0, float32);


    // read shape
    std::ifstream shape_in("/Users/scroy/repos/mlx-exp/whisper_encoder_input_shape.txt");
    int B, F, T;
    shape_in >> B >> F >> T;

    // read raw floats
    std::ifstream bin_in("/Users/scroy/repos/mlx-exp/whisper_encoder_input.bin", std::ios::binary);
    std::vector<float> buf(B * F * T);
    bin_in.read(reinterpret_cast<char*>(buf.data()),
                buf.size() * sizeof(float));

    // wrap into MLX array with shape {B, F, T}
    array encoder_in(buf.data(), Shape{B, F, T}, float32);
    eval(encoder_in);
    S_encoder.tensor_ref(encoder_input_tid) = encoder_in;
    std::cout << "Encoder in: " << encoder_in << std::endl;







    // Get encoder output
    if (P_encoder.num_outputs() != 1) {
      throw std::runtime_error("encoder program must have 1 output");
    }
    auto encoder_out_tid = std::get<Tid>(P_encoder.output_map[0]);

    std::cout << "Running encoder\n";
    auto t0p = std::chrono::steady_clock::now();
    I.run(P_encoder, S_encoder);
    const auto& encoder_output = S_encoder.const_tensor_ref(encoder_out_tid);
    eval(encoder_output); // Include in prefill timing
    auto t1p = std::chrono::steady_clock::now();

    double encoder_ms = std::chrono::duration<double, std::milli>(t1p - t0p).count();
    std::cout << "Encoder time: " << encoder_ms << " ms\n";

    // -----------------------------------------------------------------
    // Load decoder program
    // -----------------------------------------------------------------
    std::ifstream jf_dec(decoder_prog_json);
    if (!jf_dec) throw std::runtime_error("cannot open prog.json: " + decoder_prog_json);
    nlohmann::json j_dec;
    jf_dec >> j_dec;
    Program P_decoder = program_from_json(j_dec);

    ConstantData store_decoder;
    bind_constants_from_safetensors(decoder_consts_path, P_decoder, store_decoder);

    ExecutionState S_decoder;
    S_decoder.bind(P_decoder);
    init_execution_state_from_meta(P_decoder, S_decoder);

    std::cout << "[decoder] constant tensors: " << P_decoder.num_constant_tensors << "\n";
    std::cout << "[decoder] non-constant tensors: " << P_decoder.num_non_constant_tensors << "\n";
    std::cout << "[decoder] non-constant values: " << P_decoder.num_non_constant_values << "\n";
    std::cout << "[decoder] inputs: " << P_decoder.num_inputs()
              << ", outputs: " << P_decoder.num_outputs() << "\n";

    // Eval decoder constants once
    for (size_t i = 0; i < P_decoder.constants->tensors.size(); ++i) {
      eval(P_decoder.constants->tensors.at(i));
    }

    // We expect the decoder signature to look like:
    //   0: decoder_input_ids  (Tid)   [B, T_dec]
    //   1: encoder_hidden     (Tid)   [B, T_enc, D]
    //   2: input_pos          (Vid<int32_t>) scalar
    //   output 0: logits      (Tid)   [B, T_dec, V]
    if (P_decoder.num_inputs() != 3) {
      throw std::runtime_error("decoder program must have 3 inputs (ids, enc_out, input_pos)");
    }
    if (P_decoder.num_outputs() != 1) {
      throw std::runtime_error("decoder program must have 1 output (logits)");
    }

    auto decoder_ids_tid      = std::get<Tid>(P_decoder.input_map[0]);
    auto decoder_enc_out_tid  = std::get<Tid>(P_decoder.input_map[1]);
    auto decoder_inputpos_vid = std::get<Vid<int32_t>>(P_decoder.input_map[2]);
    auto decoder_logits_tid   = std::get<Tid>(P_decoder.output_map[0]);

    // Helper to set scalar input_pos WITHOUT value_ref (like Llama script)
    auto set_input_pos = [&](int32_t v) {
      if (decoder_inputpos_vid.idx >= S_decoder.values.size()) {
        throw std::out_of_range("set_input_pos: id out of range");
      }
      S_decoder.values[decoder_inputpos_vid.idx] = Value{ v };
    };

    // Bind encoder_hidden_states once: reuse encoder_output array directly
    S_decoder.tensor_ref(decoder_enc_out_tid) = encoder_output;

    // -----------------------------------------------------------------
    // Decoder loop: feed BOS, then step token-by-token with KV cache
    // -----------------------------------------------------------------
    std::cout << "Running decoder loop for " << max_new_tokens << " tokens\n";

    // Current token ids on device: start with [B=1, T=1] = BOS
    array cur_ids = full(Shape{1, 1}, bos_token_id, int32);

    // Track generated token IDs on host
    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    auto t0d = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens; ++step) {
      // Set decoder inputs
      S_decoder.tensor_ref(decoder_ids_tid) = cur_ids;
      set_input_pos(step);  // initialize/update scalar value slot

      // Run decoder step
      I.run(P_decoder, S_decoder);

      const auto& logits = S_decoder.const_tensor_ref(decoder_logits_tid);
      eval(logits);

      // Sample next token (argmax over last time step)
      array next_ids = sample_next_token(logits);  // [1,1] int32
      eval(next_ids);

      int tok_id = next_ids.item<int>();
      generated.push_back(tok_id);

      // Feed back for next iteration
      cur_ids = full(Shape{1, 1}, tok_id, int32);

      if ((step + 1) % print_batch == 0) {
        std::cout << "  generated " << (step + 1) << " tokens; last id=" << tok_id << "\n";
      }
    }
    auto t1d = std::chrono::steady_clock::now();

    double decoder_ms = std::chrono::duration<double, std::milli>(t1d - t0d).count();
    double tok_per_s  = (decoder_ms > 0.0)
                      ? (generated.size() / (decoder_ms / 1000.0))
                      : 0.0;

    std::cout << "Decoder time: " << decoder_ms << " ms for "
              << generated.size() << " tokens ("
              << tok_per_s << " tok/s)\n";

    std::cout << "Generated " << generated.size() << " tokens\n";
    std::cout << "Generated tokens: ";
    for (size_t i = 0; i < generated.size(); ++i) {
      if (i) std::cout << ' ';
      std::cout << generated[i];
    }
    std::cout << "\n";

    // ---------------- write output ids if requested ----------------
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
