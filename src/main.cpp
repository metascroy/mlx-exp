// main.cpp — name-bound initialization + refs
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
#include <optional>
#include <algorithm>
#include <cmath>

#include <mlx/ops.h>
#include <mlx/memory.h>
#include <mlx/transforms.h>

using namespace executorch::mlx;
using namespace mlx::core;

static inline Shape shape(std::initializer_list<int> v) { return Shape(v.begin(), v.end()); }

static std::vector<int> read_token_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open tokens file: " + path);
  std::vector<int> ids; ids.reserve(8192);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int x; while (iss >> x) ids.push_back(x);
  }
  if (ids.empty()) throw std::runtime_error("no token ids in: " + path);
  return ids;
}
static int env_or_int(const char* k, int d){ const char* v=getenv(k); return v? std::max(0,atoi(v)) : d; }
static std::string env_or(const char* k, const char* d){ const char* v=getenv(k); return v? std::string(v):std::string(d); }

// logits: [B, T, V] -> argmax over last step -> [B, 1] int32
static array sample_next_token(const array& logits) {
  auto s = logits.shape();
  if (s.size()!=3) throw std::runtime_error("logits must be [B,T,V]");
  int B=s[0], T=s[1], V=s[2];
  array last = (T>1) ? slice(logits, Shape{0,T-1,0}, Shape{B,T,V}) : logits;
  array idx  = argmax(last, /*axis=*/2);
  if (idx.dtype()!=int32) idx = astype(idx, int32);
  return idx; // [B,1]
}

int main() {
  try {
    const std::string model_pt  = env_or("MODEL_PT",   "model.safetensors");
    const std::string prompt_fp = env_or("PROMPT_IDS", "prompt_ids.txt");
    const int max_new_tokens    = env_or_int("MAX_NEW_TOKENS", 64);
    const int print_batch       = std::max(1, env_or_int("PRINT_BATCH", 1));

    set_wired_limit(4L * (1<<30));

    // -------- Config --------
    LlamaCfg cfg;
    cfg.B=1; cfg.T_max=4096;
    cfg.H=32; cfg.H_kv=8; cfg.D_model=2048; cfg.D_head=64;
    cfg.n_layers=16; cfg.d_ff=8192; cfg.vocab=128256;
    cfg.rope_traditional=false; cfg.rope_theta=500000.f; cfg.rms_eps=1e-6f;

    Program P;
    ConstantData consts;  // external constant storage

    // 1) Load constants (fills consts; bumps P.num_constant_tensors; binds to Program)
    auto W = load_llama_weights_from_torch(model_pt, P, consts, cfg, /*tie_lm_head=*/true);

    // 2) Graph/caches (append non-constant slots: tensors + scalar values)
    auto Cc = make_caches(P, cfg);
    auto G  = make_graph_ids(P, cfg);

    // 3) Bind helpful names (for name-based access)
    P.bind_name(G.input_ids, "input_ids");
    P.bind_name(G.logits,    "logits");
    for (int l=0;l<cfg.n_layers;++l) {
      P.bind_name(Cc.by_layer[l].K_cache, "L"+std::to_string(l)+".K_cache");
      P.bind_name(Cc.by_layer[l].V_cache, "L"+std::to_string(l)+".V_cache");
    }
    // Bind shared scalar Vids (all layers share the same Vid ids)
    {
      const auto& c0 = Cc.by_layer[0];
      P.bind_name(c0.cursor,   "cursor");
      P.bind_name(c0.sh_axis,  "sh_axis");
      P.bind_name(c0.sh_start, "sh_start");
      P.bind_name(c0.sh_len,   "sh_len");
      P.bind_name(c0.rd_axis,  "rd_axis");
      P.bind_name(c0.rd_start, "rd_start");
      P.bind_name(c0.rd_len,   "rd_len");
    }

    // 4) Emit instructions
    build_llama_shared(P, cfg, W, Cc, G);

    // -------- Runtime state --------
    ExecutionState S; S.bind(P);

    // Name-based initializers (initialize optionals safely by name)
    auto set_tensor_by_name = [&](const std::string& name, array a) {
      Tid id = std::get<Tid>(P.get_slot(name));
      // non-constant tensors are stored after constants; ExecutionState stores only the non-constant block
      size_t off = id.idx - P.num_constant_tensors;
      if (off >= S.tensors.size()) throw std::out_of_range("set_tensor_by_name: offset out of range");
      S.tensors[off] = std::move(a);
    };
    auto get_tensor_cref_by_name = [&](const std::string& name) -> const array& {
      return S.const_tensor_ref(std::get<Tid>(P.get_slot(name)));
    };
    auto get_tensor_ref_by_name = [&](const std::string& name) -> array& {
      return S.tensor_ref(std::get<Tid>(P.get_slot(name)));
    };
    auto set_value_by_name = [&](const std::string& name, int32_t v) {
      auto id = std::get<Vid<int32_t>>(P.get_slot(name));
      if (id.idx >= S.values.size()) throw std::out_of_range("set_value_by_name: id out of range");
      S.values[id.idx] = Value{ v };
    };

    // Debug layout
    std::cout << "constant tensors: " << P.num_constant_tensors << "\n";
    std::cout << "non-constant tensors: " << P.num_non_constant_tensors << "\n";
    std::cout << "non-constant values: " << P.num_non_constant_values << "\n";
    std::cout << "inputs: " << P.num_inputs() << ", outputs: " << P.num_outputs() << "\n";

    // Initialize KV caches + scalar slice helpers using name-based initializers
    {
      const Dtype act_dt = float32; // runtime dtype for cache buffers
      for (int l=0;l<cfg.n_layers;++l) {
        set_tensor_by_name("L"+std::to_string(l)+".K_cache",
          zeros(shape({cfg.B, cfg.H_kv, cfg.T_max, cfg.D_head}), act_dt));
        set_tensor_by_name("L"+std::to_string(l)+".V_cache",
          zeros(shape({cfg.B, cfg.H_kv, cfg.T_max, cfg.D_head}), act_dt));
      }
      // cursor and slice helpers (initialize values)
      set_value_by_name("cursor",   0);
      set_value_by_name("sh_axis",  2);
      set_value_by_name("rd_axis",  2);
      set_value_by_name("sh_start", 0);
      set_value_by_name("sh_len",   0);
      set_value_by_name("rd_start", 0);
      set_value_by_name("rd_len",   0);
    }

    // -------- Prefill --------
    std::cout << "Starting prefill…\n";
    std::vector<int> prompt = read_token_ids(prompt_fp);
    if ((int)prompt.size() > cfg.T_max) throw std::runtime_error("prompt too long");
    const int T_prefill = (int)prompt.size();

    // tokens → inputs tensor
    {
      array ids(prompt.data(), Shape{(int)prompt.size()}, int32);
      ids = reshape(ids, Shape{cfg.B, T_prefill});
      set_tensor_by_name("input_ids", std::move(ids));
    }

    // set cursor/slices for prefill: write/read [0, T_prefill)
    {
      set_value_by_name("cursor",   0);
      set_value_by_name("sh_start", 0);
      set_value_by_name("sh_len",   T_prefill);
      set_value_by_name("rd_start", 0);
      set_value_by_name("rd_len",   T_prefill);
    }

    Interpreter I;

    std::cout << "Running prefill on T=" << T_prefill << "…\n";
    auto t0p = std::chrono::steady_clock::now();
    I.run(P, S);
    array next_ids = sample_next_token(get_tensor_cref_by_name("logits"));
    eval(next_ids);
    auto t1p = std::chrono::steady_clock::now();
    std::cout << "Prefill token: " << next_ids.item<int>() << "\n";
    std::cout << "Prefill time: "
              << std::chrono::duration<double, std::milli>(t1p-t0p).count() << " ms\n";

    // -------- Seed decode slice helpers (write one step at T_prefill) --------
    {
      set_value_by_name("cursor",   T_prefill);
      // write exactly one new time index at `cursor`
      set_value_by_name("sh_start", T_prefill);
      set_value_by_name("sh_len",   1);
      // read covers [0, T_prefill)
      set_value_by_name("rd_start", 0);
      set_value_by_name("rd_len",   T_prefill);
    }

    // -------- Decode --------
    array cur_ids = next_ids; // [B,1]
    int cursor = T_prefill;
    std::vector<int> printed; printed.reserve(max_new_tokens);

    auto t0d = std::chrono::steady_clock::now();
    int batch_count = 0;
    std::vector<array> pending; pending.reserve(print_batch);

    for (int step=0; step<max_new_tokens; ++step) {
      set_tensor_by_name("input_ids", cur_ids);

      int new_cursor = cursor + 1;

      set_value_by_name("cursor",   cursor);
      // Write [cursor, cursor+1)
      set_value_by_name("sh_start", cursor);
      set_value_by_name("sh_len",   1);
      // Read [0, new_cursor)
      set_value_by_name("rd_start", 0);
      set_value_by_name("rd_len",   new_cursor);

      I.run(P, S);

      array next_ids2 = sample_next_token(get_tensor_cref_by_name("logits"));
      async_eval(next_ids2);

      pending.push_back(next_ids2);
      ++batch_count;

      if (batch_count == print_batch) {
        std::cout << "[tokens " << (int)printed.size()+1 << "-" << (int)printed.size()+batch_count << "]: ";
        for (int i=0;i<batch_count;++i) {
          int t = pending[i].item<int>();
          if (i) std::cout << ' ';
          std::cout << t;
          printed.push_back(t);
        }
        std::cout << "\n";
        pending.clear(); batch_count = 0;
      }

      cur_ids = std::move(next_ids2);
      cursor = new_cursor;
      if (cursor >= cfg.T_max) break;
    }

    if (!pending.empty()) {
      std::cout << "[tokens " << (int)printed.size()+1 << "-" << (int)printed.size()+pending.size() << "]: ";
      for (size_t i=0;i<pending.size();++i) {
        int t = pending[i].item<int>();
        if (i) std::cout << ' ';
        std::cout << t;
      }
      std::cout << "\n";
      pending.clear();
    }

    synchronize();
    auto t1d = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1d-t0d).count();
    double tps = printed.size() / std::max(ms/1000.0, 1e-9);
    std::cout << "Generated " << printed.size() << " tokens\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << tps << " tok/s\n";

    std::cout << "Generated tokens: ";
    for (int t : printed) std::cout << t << ' ';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
