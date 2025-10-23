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
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

using namespace executorch::mlx;
using namespace mlx::core;

// Helpers
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

int main(int, char**) {
  try {
    // ---------------- Inputs via env ----------------
    const std::string model_pt  = env_or("MODEL_PT",   "model.pt");
    const std::string prompt_fp = env_or("PROMPT_IDS", "prompt_ids.txt");
    const int max_new_tokens    = env_or_int("MAX_NEW_TOKENS", 64);

    // ---------------- Config (must match checkpoint) ----------------
    LlamaCfg cfg;
    cfg.B        = 1;
    cfg.T_max    = 4096;     // beware KV memory
    cfg.H        = 32;
    cfg.H_kv     = 8;        // set to H if no GQA support
    cfg.D_model  = 2048;
    cfg.D_head   = 64;
    cfg.n_layers = 16;
    cfg.d_ff     = 8192;
    cfg.vocab    = 128256;

    // ---------------- Build graph & state ----------------
    Program prog;            // shared program (prefill + decode)
    MutableData st;          // runtime state: inputs, KV tensors, cursor/shapes

    // Load weights into ConstantData
    ConstantData cdata;
    auto W = load_llama_weights_from_torch(model_pt, cdata, cfg, /*tie_lm_head_to_tok_emb=*/true);

    // KV caches (ids + state)
    auto C = make_llama_cache_ids(cfg, /*base_mid=*/10000);
    init_llama_cache_state(st, cfg, C);

    // IO MIDs
    Mid input_ids{0};
    Mid logits{1};

    // Build graph
    auto G = build_llama_shared(prog, cfg, W, C, input_ids, logits);
    prog.C = std::make_shared<const ConstantData>(std::move(cdata));

    // Quick embedding sanity check
    {
      const auto& E = prog.C->c_ref(W.tok_emb);
      auto Ef = astype(E, float32);
      auto nfin = sum(astype(isfinite(Ef), float32)); nfin.eval();
      double total = 1.0; for (int d : E.shape()) total *= d;
      std::cout << "[sanity] tok_emb finite=" << nfin.item<float>() << " of " << total << "\n";
      if (nfin.item<float>() < total) throw std::runtime_error("NaNs in tok_emb: bad state_dict import");
    }

    Interpreter interp;

    // ---------------- Prefill with prompt ----------------
    std::vector<int> prompt_ids = read_token_ids(prompt_fp);
    if ((int)prompt_ids.size() > cfg.T_max) {
      throw std::runtime_error("prompt too long for T_max");
    }

    // [B, T_prompt] int32
    const int T_prefill = (int)prompt_ids.size();
    array ids_arr(prompt_ids.data(), Shape{(int)prompt_ids.size()}, int32);
    ids_arr = reshape(ids_arr, {cfg.B, T_prefill});
    st.set_mutable_id(input_ids, std::move(ids_arr));

    // Window [0 : T_prefill] with cursor=0 for RoPE during prefill
    set_prefill_cursor_shared(st, cfg, C, /*T_written=*/T_prefill);

    std::cout << "Running prefill on T=" << T_prefill << "...\n";
    auto t0 = std::chrono::steady_clock::now();
    interp.run(prog, st);
    mlx::core::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::fixed << std::setprecision(3)
              << "Prefill time: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // ---------------- Handoff to decode ----------------
    begin_decode_after_prefill(st, cfg, C, T_prefill);

    // ---------------- Decode (greedy) ----------------
    std::vector<int> generated; generated.reserve(max_new_tokens);

// Device buffer to collect all generated ids: [B, max_new_tokens] (i32)
array gen_buf = zeros({cfg.B, max_new_tokens}, int32);

// Optional: ban special/control tokens
const int ban_from_id = 128000;
const bool ban_special = true;
auto mask_special_logits_inplace = [&](array& last_logits /*[B,1,V]*/) {
  if (!ban_special) return;
  const auto s = last_logits.shape();  // [B,1,V]
  const int V = s[2];
  if (ban_from_id < V) {
    std::vector<int> idx(V);
    for (int j = 0; j < V; ++j) idx[j] = (j >= ban_from_id) ? 1 : 0;
    array mask_v(idx.data(), Shape{V}, int32);
    mask_v = astype(mask_v, float32);
    last_logits = add(last_logits, reshape(mask_v, {1,1,V}) * (-1e30f));
  }
};

double compute_ms = 0.0;
auto t0_all = std::chrono::steady_clock::now();

for (int step = 0; step < max_new_tokens; ++step) {
  const int cursor = st.i32_ref(C.layer_cache[0].cursor);
  if (cursor <= 0) throw std::runtime_error("cursor must be > 0 before decode");
  if (cursor >= cfg.T_max) {
    std::cerr << "[i] Reached T_max, stopping decode.\n";
    break;
  }

  // ---- pick next token on device ----
  array logits_ref = st.m_ref(logits);                // [B,T,V]
  auto s = logits_ref.shape();
  const int B = s[0], Tdim = s[1], V = s[2];
  const int last_idx = (Tdim == 1) ? 0 : std::max(0, cursor - 1);
  array last_logits = (Tdim == 1)
      ? logits_ref
      : slice(logits_ref, Shape{0, last_idx, 0}, Shape{B, last_idx + 1, V});

  mask_special_logits_inplace(last_logits);

  array next_ids = astype(argmax(last_logits, /*axis=*/2), int32); // [B,1]

  // ---- write into device buffer at column 'step' ----
  // gen_buf[:, step:step+1] = next_ids
  gen_buf = slice_update(gen_buf,
                         next_ids,
                         Shape{0, step},    // start
                         Shape{B, step+1}); // stop

  // ---- feed token and run one decode step ----
  st.set_mutable_id(input_ids, next_ids);

  // Advance KV cursor/windows by 1 (do NOT sync)
  advance_decode_cursor_shared(st, cfg, C, cursor, /*t_step=*/1);

  auto t0s = std::chrono::steady_clock::now();
  interp.run(prog, st);  // if this internally syncs, that limits TPS; we still avoid host copies
  mlx::core::synchronize();
  auto t1s = std::chrono::steady_clock::now();
  compute_ms += std::chrono::duration<double, std::milli>(t1s - t0s).count();
}

// ---- One final sync + host transfer for the whole sequence ----
mlx::core::synchronize();

// gen_buf: [B, max_new_tokens] -> squeeze to [max_new_tokens] (B==1), copy once, then read
array gen_host = copy(reshape(gen_buf, {max_new_tokens}));

generated.resize(max_new_tokens);
for (int i = 0; i < max_new_tokens; ++i) {
  // Take [i:i+1] slice -> reshape to scalar -> read item<int>()
  array s = slice(gen_host, Shape{i}, Shape{i + 1});
  int v = copy(reshape(s, {})).item<int>();
  generated[i] = v;
}

auto t1_all = std::chrono::steady_clock::now();
double wall_ms = std::chrono::duration<double, std::milli>(t1_all - t0_all).count();

std::cout << "\nGenerated token ids (" << generated.size() << "): ";
for (size_t i = 0; i < generated.size(); ++i) {
  if (i) std::cout << ' ';
  std::cout << generated[i];
}
std::cout << "\n";

if (!generated.empty()) {
  double tps_compute = generated.size() / std::max(compute_ms / 1000.0, 1e-9);
  double tps_wall    = generated.size() / std::max(wall_ms    / 1000.0, 1e-9);
  std::cout << "Throughput (compute): " << tps_compute << " tok/s\n";
  std::cout << "Throughput (wall):    " << tps_wall    << " tok/s\n";
}
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
