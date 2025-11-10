// llm_entry.cpp — JSON + safetensors + prompt runner for iOS
#include "llm_entry.hpp"
#include "ops.hpp"
#include "program.hpp"
#include "interpreter.hpp"
#include "program_json_loader.hpp"

#include <mlx/ops.h>
#include <mlx/memory.h>
#include <mlx/transforms.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <iostream>

using namespace executorch::mlx;
using namespace mlx::core;

static void log_msg(void (*on_log)(const char*, void*), void* user, const std::string& s) {
  if (on_log) on_log(s.c_str(), user);
}

// read space/newline separated token ids
static std::vector<int> read_token_ids_path(const std::string& path) {
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

// logits: [B,T,V] -> argmax last step -> [B,1] int32
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

extern "C"
int run_llm_generation(const char* prog_json_path_c,
                       const char* consts_path_c,
                       const char* prompt_ids_path_c,
                       int max_new_tokens,
                       int print_batch,
                       void (*on_tokens)(const int* ids, int count, void* user),
                       void* user,
                       void (*on_log)(const char* msg, void* user)) {
  try {
    const std::string prog_json_path = prog_json_path_c ? prog_json_path_c : "";
    const std::string consts_path    = consts_path_c    ? consts_path_c    : "";
    const std::string prompt_path    = prompt_ids_path_c? prompt_ids_path_c: "";
    if (prog_json_path.empty())  throw std::runtime_error("missing prog.json path");
    if (consts_path.empty())     throw std::runtime_error("missing consts.safetensors path");
    if (prompt_path.empty())     throw std::runtime_error("missing prompt_ids.txt path");
    if (print_batch < 1) print_batch = 1;

    // load program JSON
    {
      std::ostringstream oss;
      oss << "Loading program: " << prog_json_path;
      log_msg(on_log, user, oss.str());
    }
    std::ifstream jf(prog_json_path);
    if (!jf) throw std::runtime_error("cannot open prog.json: " + prog_json_path);
    nlohmann::json j;
    jf >> j;
    Program P = program_from_json(j);

    // load constants (safetensors)
    ConstantData store;
    bind_constants_from_safetensors(consts_path, P, store);

    // execution state
    ExecutionState S;
    S.bind(P);
    init_execution_state_from_meta(P, S);

    // helpers
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

    // read prompt
    const std::vector<int> prompt = read_token_ids_path(prompt_path);
    const int T_prefill = (int)prompt.size();
    if (T_prefill <= 0) throw std::runtime_error("prompt is empty");

    // name assumptions from your JSON pipeline:
    // - input token tensor name: "token_ids"
    // - position value name:     "input_pos"
    // - logits tensor name:      "linear_112" (change if your export differs)
    {
      array ids(prompt.data(), Shape{T_prefill}, int32);
      ids = reshape(ids, Shape{1, T_prefill});
      eval(ids);
      set_tensor_by_name("token_ids", std::move(ids));
    }
    set_value_by_name("input_pos", 0);

    // eval constants before timing
    for (size_t i = 0; i < P.constants->tensors.size(); ++i) {
      eval(P.constants->tensors.at(i));
    }

    set_wired_limit(4L * (1 << 30)); // 4GB
    Interpreter I;
      
    assert(P.num_outputs() == 1);
    auto out_tid = std::get<Tid>(P.output_map[0]);
      
    // ---------------- prefill ----------------
    log_msg(on_log, user, "Start prefill");
    auto t0p = std::chrono::steady_clock::now();
    I.run(P, S);

    const auto& logits_prefill = S.const_tensor_ref(out_tid);
    array next_ids = sample_next_token(logits_prefill);
    eval(next_ids);
    int prefill_token = next_ids.item<int>();
    auto t1p = std::chrono::steady_clock::now();
    double prefill_time_ms = std::chrono::duration<double,std::milli>(t1p-t0p).count();
    double prefill_tok_per_sec = T_prefill / (prefill_time_ms / 1000.0);

    log_msg(on_log, user, "Prefill token: " + std::to_string(prefill_token));
    log_msg(on_log, user, "Prefill time (ms): " + std::to_string(prefill_time_ms));
    log_msg(on_log, user, "Prefill tokens: " + std::to_string(T_prefill));
    log_msg(on_log, user, "Prefill tok/sec: " + std::to_string(prefill_tok_per_sec));

    // ---------------- decode ----------------
    array cur_ids = next_ids;
    int cursor = T_prefill;
    std::vector<int> batch; batch.reserve(std::max(1, print_batch));
    auto t0d = std::chrono::steady_clock::now();

    for (int step = 0; step < max_new_tokens; ++step) {
      set_tensor_by_name("token_ids", cur_ids);
      set_value_by_name("input_pos", cursor);

      I.run(P, S);
      if (step == 0) {
        clear_cache();
      }

      const auto& logits = S.const_tensor_ref(out_tid);
      array next_ids2 = sample_next_token(logits);
      async_eval(next_ids2);

      int t = next_ids2.item<int>();
      batch.push_back(t);
      if ((int)batch.size() == print_batch) {
        if (on_tokens) on_tokens(batch.data(), (int)batch.size(), user);
        batch.clear();
      }

      cur_ids = std::move(next_ids2);
      ++cursor;
    }

    if (!batch.empty() && on_tokens) {
      on_tokens(batch.data(), (int)batch.size(), user);
    }
    synchronize();

    auto t1d = std::chrono::steady_clock::now();
    const double decode_time_ms = std::chrono::duration<double,std::milli>(t1d-t0d).count();
    log_msg(on_log, user, ("Decode ms: " + std::to_string(decode_time_ms)));
    log_msg(on_log, user, ("Decode tokens: " + std::to_string(max_new_tokens)));
    double decode_tps = max_new_tokens / (decode_time_ms / 1000.0);
    log_msg(on_log, user, ("Decode tok/sec: " + std::to_string(decode_tps)));

    return 0;
  } catch (const std::exception& e) {
    if (on_log) on_log(e.what(), user);
    return 1;
  }
}
