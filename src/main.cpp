// main2.cpp — run exported JSON program with existing Interpreter
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
// Token file helper
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
  return idx;
}

int main() {
  try {
    const std::string prog_json   = env_or("PROG_JSON",   "/Users/scroy/repos/mlx-exp/prog.json");
    const std::string consts_path = env_or("CONSTS_ST",   "/Users/scroy/repos/mlx-exp/consts.safetensors");
    const std::string prompt_fp   = env_or("PROMPT_IDS",  "/Users/scroy/repos/mlx-exp/prompt_ids.txt");
    const int max_new_tokens      = env_or_int("MAX_NEW_TOKENS", 128);
    const int print_batch         = std::max(1, env_or_int("PRINT_BATCH", 1));
    const std::string output_ids  = env_or("OUTPUT_IDS",  "/Users/scroy/repos/mlx-exp/output_ids.txt");

    set_wired_limit(4L * (1 << 30));  // 4GB

    // load program
    std::ifstream jf(prog_json);
    if (!jf) throw std::runtime_error("cannot open prog.json: " + prog_json);
    nlohmann::json j;
    jf >> j;
    Program P = program_from_json(j);

    // load constants
    ConstantData store;
    bind_constants_from_safetensors(consts_path, P, store);

    // Execution state
    ExecutionState S;
    S.bind(P);
    init_execution_state_from_meta(P, S);

    // name helpers
    auto set_tensor_by_name = [&](const std::string& name, array a) {
      Tid id = std::get<Tid>(P.get_slot(name));
      size_t off = id.idx - P.num_constant_tensors;
      if (off >= S.tensors.size())
        throw std::out_of_range("set_tensor_by_name: offset out of range");
      S.tensors[off] = std::move(a);
    };
    auto get_tensor_cref_by_name = [&](const std::string& name) -> const array& {
      return S.const_tensor_ref(std::get<Tid>(P.get_slot(name)));
    };
    auto set_value_by_name = [&](const std::string& name, int32_t v) {
      auto id = std::get<Vid<int32_t>>(P.get_slot(name));
      if (id.idx >= S.values.size())
        throw std::out_of_range("set_value_by_name: id out of range");
      S.values[id.idx] = Value{ v };
    };

    std::cout << "constant tensors: " << P.num_constant_tensors << "\n";
    std::cout << "non-constant tensors: " << P.num_non_constant_tensors << "\n";
    std::cout << "non-constant values: " << P.num_non_constant_values << "\n";
    std::cout << "inputs: " << P.num_inputs() << ", outputs: " << P.num_outputs() << "\n";

    // prompt → token_ids
    std::vector<int> prompt = read_token_ids(prompt_fp);
    const int T_prefill = (int)prompt.size();
    if (T_prefill <= 0) throw std::runtime_error("prompt is empty");

    {
      array ids(prompt.data(), Shape{T_prefill}, int32);
      ids = reshape(ids, Shape{1, T_prefill});
      eval(ids);
      set_tensor_by_name("token_ids", std::move(ids));
    }
    set_value_by_name("input_pos", 0);

    // Eval constants before timing prefill
    for (size_t i = 0; i < P.constants->tensors.size(); ++i) {
      eval(P.constants->tensors.at(i));
    }

    Interpreter I;

    // ---------------- prefill ----------------
    std::cout << "Running prefill on T=" << T_prefill << "…\n";
    auto t0p = std::chrono::steady_clock::now();
    I.run(P, S);

    const auto& logits_prefill = get_tensor_cref_by_name("linear_112");
    array next_ids = sample_next_token(logits_prefill);
    eval(next_ids); // Include in prefill timing
    auto t1p = std::chrono::steady_clock::now();

    int next_id = next_ids.item<int>();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1p - t0p).count();
    double prefill_tps = T_prefill / std::max(prefill_ms / 1000.0, 1e-9);
    std::cout << "Prefill token: " << next_id << "\n";
    std::cout << "Prefill time: " << prefill_ms << " ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << prefill_tps << " tok/s\n";

    // ---------------- first decode step ----------------
    int cursor = T_prefill;
    {
      array tok = full(Shape{1,1}, next_id, int32);
      set_tensor_by_name("token_ids", std::move(tok));
      set_value_by_name("input_pos", cursor);
      I.run(P, S);
    }

    const auto& logits1 = get_tensor_cref_by_name("linear_112");
    array next_ids2 = sample_next_token(logits1);
    async_eval(next_ids2);   // schedule
    int tok_id = next_ids2.item<int>();

    // we already have two tokens, print them right away
    std::vector<int> printed;
    printed.reserve(max_new_tokens);
    printed.push_back(next_id);  // from prefill
    printed.push_back(tok_id);   // from first decode

    next_id = tok_id;
    ++cursor;

    // ---------------- remaining decode ----------------
    std::vector<array> pending;
    pending.reserve(print_batch);

    auto t0d = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
      // feed token
      array tok = full(Shape{1,1}, next_id, int32);
      set_tensor_by_name("token_ids", std::move(tok));
      set_value_by_name("input_pos", cursor);
      I.run(P, S);

      // get logits, schedule next, store for print
      const auto& logits = get_tensor_cref_by_name("linear_112");
      array nx = sample_next_token(logits);
      async_eval(nx);
      pending.push_back(nx);

      // read this token now for the next iteration
      int t = nx.item<int>();
      next_id = t;
      ++cursor;

      // if we filled the batch, print them
      if ((int)pending.size() == print_batch) {
        std::cout << "[tokens " << printed.size()+1
                  << "-" << printed.size()+pending.size() << "]: ";
        for (size_t i = 0; i < pending.size(); ++i) {
          int tt = pending[i].item<int>();
          if (i) std::cout << ' ';
          std::cout << tt;
          printed.push_back(tt);
        }
        std::cout << "\n";
        pending.clear();
      }
    }

    // flush leftover (if decode ended before filling a batch)
    if (!pending.empty()) {
      std::cout << "[tokens " << printed.size()+1
                << "-" << printed.size()+pending.size() << "]: ";
      for (size_t i = 0; i < pending.size(); ++i) {
        int tt = pending[i].item<int>();
        if (i) std::cout << ' ';
        std::cout << tt;
        printed.push_back(tt);
      }
      std::cout << "\n";
      pending.clear();
    }

    synchronize();
    auto t1d = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t1d - t0d).count();
    double decode_tps = printed.size() / std::max(decode_ms / 1000.0, 1e-9);

    std::cout << "Generated " << printed.size() << " tokens\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << decode_tps << " tok/s\n";
    std::cout << "Generated tokens: ";
    for (int t : printed) std::cout << t << ' ';
    std::cout << "\n";

    // ---------------- write output ids if requested ----------------
    if (!output_ids.empty()) {
      std::ofstream ofs(output_ids);
      if (!ofs) {
        std::cerr << "warning: could not open OUTPUT_IDS file: " << output_ids << "\n";
      } else {
        for (size_t i = 0; i < printed.size(); ++i) {
          if (i) ofs << ' ';
          ofs << printed[i];
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
