// main.cpp
#include "id.hpp"
#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"
#include "llm_builder.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <optional>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <cstring> // memcpy

#include <mlx/transforms.h>
#include <mlx/memory.h>
#include <mlx/ops.h>  // added: default_device(), new_stream(), synchronize(stream)

using namespace executorch::mlx;
using namespace mlx::core;

// -----------------------------
// Helpers
// -----------------------------
static std::vector<int> read_token_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open tokens file: " + path);
  std::vector<int> ids; ids.reserve(4096);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int x; while (iss >> x) ids.push_back(x);
  }
  if (ids.empty()) throw std::runtime_error("no token ids found in: " + path);
  return ids;
}

static int env_or_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::max(0, std::atoi(v)) : def;
}

static std::string env_or(const char* name, const char* def) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string(def);
}

// -------------------------------------------------------
// Sampling: argmax over vocab, return [B,1] int32 tokens
// logits: [B, T, V]  (prefill T>=1, decode T==1 usually)
// -------------------------------------------------------
static inline mlx::core::array sample_next_token(const mlx::core::array& logits,
                                                 bool /*is_prefill*/) {
  using namespace mlx::core;
  auto s = logits.shape();
  if (s.size() != 3) {
    throw std::runtime_error("sample_next_token: expected logits rank-3 [B,T,V]");
  }
  const int B = s[0], T = s[1], V = s[2];

  array last_logits = logits;
  if (T > 1) {
    // [B,1,V] slice at last timestep
    last_logits = slice(logits, Shape{0, T - 1, 0}, Shape{B, T, V});
  }

  // Argmax over vocab -> [B,1] (int32)
  array next_ids = astype(argmax(last_logits, /*axis=*/2), int32);
  return next_ids; // caller controls evaluation/sync
}

int main(int, char**) {
  try {
    // ---------------- Inputs via env ----------------
    const std::string model_pt   = env_or("MODEL_PT",   "model.pt");
    const std::string prompt_fp  = env_or("PROMPT_IDS", "prompt_ids.txt");
    const int max_new_tokens     = env_or_int("MAX_NEW_TOKENS", 64);
    const int print_batch        = std::max(1, env_or_int("PRINT_BATCH", 1)); // configurable "8"

    // ---------------- Config ----------------
    LlamaCfg cfg;
    cfg.B        = 1;
    cfg.T_max    = 4096;
    cfg.H        = 32;
    cfg.H_kv     = 8;
    cfg.D_model  = 2048;
    cfg.D_head   = 64;
    cfg.n_layers = 16;
    cfg.d_ff     = 8192;
    cfg.vocab    = 128256;

    // ---------------- Build graph ----------------
    Program prog;
    MutableData st;
    ConstantData cdata;
    auto W = load_llama_weights_from_torch(model_pt, cdata, cfg, /*tie_lm_head_to_tok_emb=*/false);
    auto C = make_llama_cache_ids(cfg, /*base_mid=*/10000);
    init_llama_cache_state(st, cfg, C);

    Mid input_ids{0};
    Mid logits{1};
    auto G = build_llama_shared(prog, cfg, W, C, input_ids, logits);
    prog.C = std::make_shared<const ConstantData>(std::move(cdata));

    // Sanity check
    {
      const auto& E = prog.C->c_ref(W.tok_emb);
      auto Ef = astype(E, float32);
      auto nfin = sum(astype(isfinite(Ef), float32)); nfin.eval();
      double total = 1.0; for (int d : E.shape()) total *= d;
      std::cout << "[sanity] tok_emb finite=" << nfin.item<float>() << " of " << total << "\n";
      if (nfin.item<float>() < total) throw std::runtime_error("NaNs in tok_emb: bad state_dict import");
    }

    Interpreter interp;

    // ---- Create a dedicated stream for generation (added) ----
    // auto dev        = default_device();
    // auto gen_stream = new_stream(dev);

    // ---------------- Prefill ----------------
    std::vector<int> prompt_ids = read_token_ids(prompt_fp);
    if ((int)prompt_ids.size() > cfg.T_max) throw std::runtime_error("prompt too long");
    const int T_prefill = (int)prompt_ids.size();

    array ids_arr(prompt_ids.data(), Shape{(int)prompt_ids.size()}, int32);
    ids_arr = reshape(ids_arr, {cfg.B, T_prefill});
    st.set_mutable_id(input_ids, std::move(ids_arr));

    set_prefill_cursor_shared(st, cfg, C, /*T_written=*/T_prefill);
    std::cout << "Running prefill on T=" << T_prefill << "...\n";

    auto t0_prefill = std::chrono::steady_clock::now();
    interp.run(prog, st); // changed: pass stream

    // Sample next token from prefill logits via argmax
    array next_ids_prefill = sample_next_token(st.m_ref(logits), /*is_prefill=*/true);

    // ---- Synchronized evaluation at end of prefill ----
    mlx::core::eval(next_ids_prefill);

    std::cout << "Prefill token: " << next_ids_prefill.item<int>() << "\n";

    auto t1_prefill = std::chrono::steady_clock::now();
    std::cout << std::fixed << std::setprecision(3)
              << "Prefill time: "
              << std::chrono::duration<double, std::milli>(t1_prefill - t0_prefill).count()
              << " ms\n";

    // ---------------- Decode (pipelined with batched host sync) ----------------
    begin_decode_after_prefill(st, cfg, C, T_prefill);

    // Current token to be *printed next iteration*. Async-eval it now.
    array cur_input_ids = next_ids_prefill;   // [B,1], int32
    // mlx::core::async_eval(cur_input_ids);     // ensures printed token had async_eval in prior iter

    // Cursor at next write position (== T_prefill). We advance-at-index then ++.
    int cursor = st.i32_ref(C.layer_cache[0].cursor);

    // Device-ring buffer for tokens to print (optional<> avoids default-ctor)
    std::vector<std::optional<mlx::core::array>> dev_tok_buf(print_batch);
    int dev_tok_count = 0;

    std::vector<int> generated_host; generated_host.reserve(max_new_tokens);

    auto t0_decode = std::chrono::steady_clock::now();
    int printed = 0;

    while (printed < max_new_tokens) {
      // Feed current token to produce logits for next token.
      st.set_mutable_id(input_ids, cur_input_ids);

      // Advance KV cursor for this single step at current index
      advance_decode_cursor_shared(st, cfg, C, cursor, /*len=*/1);
      cursor += 1;

      // Run one decode step
      interp.run(prog, st); // changed: pass stream

      // Sample the *next* token from these logits
      array next_ids = sample_next_token(st.m_ref(logits), /*is_prefill=*/false);

      // Pipeline: async-eval the next token now...
      mlx::core::async_eval(next_ids);

      // ...then enqueue the *current* token for batched host read/print
      dev_tok_buf[dev_tok_count++] = std::move(cur_input_ids);
      ++printed;

      // Flush in batches of print_batch (1-indexed ranges in the label)
      if (dev_tok_count == print_batch) {
        std::cout << "[tokens " << (printed - (print_batch - 1)) << "-" << printed << "]: ";
        for (int i = 0; i < print_batch; ++i) {
          // Tiny sync per token, but batched instead of every step
          int tok = dev_tok_buf[i]->item<int>();
          if (i) std::cout << ' ';
          std::cout << tok;
          generated_host.push_back(tok);
          dev_tok_buf[i].reset();
        }
        std::cout << "\n";
        dev_tok_count = 0;
      }

      // Prepare for next iteration: the "next" becomes "current"
      cur_input_ids = std::move(next_ids);

      // if (printed % 256 == 0) {
      //   mlx::core::clear_cache();
      // }

      if (cursor >= cfg.T_max) break; // safety
    }

    // Flush any remaining tokens (<print_batch tail)
    if (dev_tok_count > 0) {
      std::cout << "[tokens " << (printed - dev_tok_count + 1)
                << "-" << printed << "]: ";
      for (int i = 0; i < dev_tok_count; ++i) {
        int tok = dev_tok_buf[i]->item<int>();
        if (i) std::cout << ' ';
        std::cout << tok;
        generated_host.push_back(tok);
        dev_tok_buf[i].reset();
      }
      std::cout << "\n";
      dev_tok_count = 0;
    }

    // Ensure all outstanding device work is done before timing
    synchronize(); // changed: sync specific stream

    auto t1_decode = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1_decode - t0_decode).count();
    double tps = printed / std::max(ms / 1000.0, 1e-9);

    std::cout << "Generated " << printed << " tokens total\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << tps << " tok/s\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
