// main.cpp
#include "interpreter.hpp"
#include "ops.hpp"
#include "graph_build.hpp"

#include <iostream>
#include <random>

// Small helper to fill an MLX array with deterministic values.
static void fill_uniform(mlx::core::array& a,
                         float lo = -0.02f,
                         float hi = 0.02f,
                         uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);

  // Materialize a host buffer then copy to device (simple & portable).
  auto shape = a.shape();                 // mlx::core::Shape
  size_t n = 1;
  for (int d : shape) n *= static_cast<size_t>(d);

  std::vector<float> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = dist(rng);

  // Construct an MLX array from host data and reshape to 'shape'
  auto host_arr = mlx::core::array(host.data(),
                                   mlx::core::Shape{static_cast<int>(n)},
                                   mlx::core::float32);
  host_arr = mlx::core::reshape(host_arr, shape);
  a = mlx::core::astype(host_arr, a.dtype());
}

int main() {
  using namespace llm;

  std::cout << "MLX LLM interpreter demo (mini MLP/SwiGLU)\n";

  // ------------------------------------------------------------
  // 1) Tensor IDs (TIDs) — fixed slots in the tensor table
  // ------------------------------------------------------------
  enum : Tid {
    T_X = 0,      // [B, D]
    T_W_UP,       // [D, H]
    T_W_GATE,     // [D, H]
    T_W_DOWN,     // [H, D]
    T_UP,         // [B, H]
    T_GATE,       // [B, H]
    T_ACT,        // [B, H]
    T_H,          // [B, H]
    T_MLP_OUT,    // [B, D]
    T_Y,          // [B, D]  (residual add back into X)
    T__COUNT
  };

  // Shape params for a toy example
  const int B = 2;       // batch
  const int D = 8;       // model dim
  const int H = 16;      // hidden dim (MLP expansion)

  // ------------------------------------------------------------
  // 2) Allocate tensor table
  // ------------------------------------------------------------
  Interpreter interp;

  // Create arrays (use interp.set so optionals are engaged)
  interp.set(T_X,       mlx::core::zeros(mlx::core::Shape{B, D}, mlx::core::float32));
  interp.set(T_W_UP,    mlx::core::zeros(mlx::core::Shape{D, H}, mlx::core::float32));
  interp.set(T_W_GATE,  mlx::core::zeros(mlx::core::Shape{D, H}, mlx::core::float32));
  interp.set(T_W_DOWN,  mlx::core::zeros(mlx::core::Shape{H, D}, mlx::core::float32));
  interp.set(T_UP,      mlx::core::zeros(mlx::core::Shape{B, H}, mlx::core::float32));
  interp.set(T_GATE,    mlx::core::zeros(mlx::core::Shape{B, H}, mlx::core::float32));
  interp.set(T_ACT,     mlx::core::zeros(mlx::core::Shape{B, H}, mlx::core::float32));
  interp.set(T_H,       mlx::core::zeros(mlx::core::Shape{B, H}, mlx::core::float32));
  interp.set(T_MLP_OUT, mlx::core::zeros(mlx::core::Shape{B, D}, mlx::core::float32));
  interp.set(T_Y,       mlx::core::zeros(mlx::core::Shape{B, D}, mlx::core::float32));

  // Initialize inputs & weights (unwrap via interp.at to get array&)
  fill_uniform(interp.at(T_X),       -0.5f, 0.5f, 123);
  fill_uniform(interp.at(T_W_UP),    -0.1f, 0.1f,  21);
  fill_uniform(interp.at(T_W_GATE),  -0.1f, 0.1f,  22);
  fill_uniform(interp.at(T_W_DOWN),  -0.1f, 0.1f,  23);

  // ------------------------------------------------------------
  // 3) Build a mini graph: SwiGLU MLP block
  //
  //   up   = X @ W_up           // [B, H]
  //   gate = X @ W_gate         // [B, H]
  //   act  = silu(gate)         // [B, H]
  //   h    = up * act           // [B, H]
  //   mlp  = h @ W_down         // [B, D]
  //   y    = X + mlp            // [B, D]   (residual)
  // ------------------------------------------------------------
  std::vector<Instr> prog;

  // up = X @ W_up
  {
    MatmulNode n{};
    n.a = T_X; n.b = T_W_UP; n.out = T_UP;
    n.ta = false; n.tb = false; n.bias = std::nullopt;
    prog.push_back(make_matmul(std::move(n)));
  }
  // gate = X @ W_gate
  {
    MatmulNode n{};
    n.a = T_X; n.b = T_W_GATE; n.out = T_GATE;
    n.ta = false; n.tb = false; n.bias = std::nullopt;
    prog.push_back(make_matmul(std::move(n)));
  }
  // act = silu(gate)
  {
    SiluNode n{};
    n.x = T_GATE; n.out = T_ACT;
    prog.push_back(make_silu(std::move(n)));
  }
  // h = up * act
  {
    MulNode n{};
    n.a = T_UP; n.b = T_ACT; n.out = T_H;
    prog.push_back(make_mul(std::move(n)));
  }
  // mlp_out = h @ W_down
  {
    MatmulNode n{};
    n.a = T_H; n.b = T_W_DOWN; n.out = T_MLP_OUT;
    n.ta = false; n.tb = false; n.bias = std::nullopt;
    prog.push_back(make_matmul(std::move(n)));
  }
  // y = X + mlp_out
  {
    AddNode n{};
    n.a = T_X; n.b = T_MLP_OUT; n.out = T_Y;
    prog.push_back(make_add(std::move(n)));
  }

  llm::BuildOptions opt;
  opt.force_argmax_i64 = true;
  opt.force_gather_i32 = true;
  opt.prebias_matmul   = true;
  opt.normalize_axes   = true;
  auto built = llm::build_and_validate(prog, opt);

  // ------------------------------------------------------------
  // 4) Run the interpreter
  // ------------------------------------------------------------
  interp.run(built.program);

  // Force materialization of the result and print a preview
  auto& Y = interp.at(T_Y);

  // Print shape
  std::cout << "Output shape: [";
  auto shp = Y.shape();
  for (size_t i = 0; i < shp.size(); ++i) {
    std::cout << shp[i] << (i + 1 < shp.size() ? ", " : "");
  }
  std::cout << "]\n";

  // Print first row (small preview)
  // NOTE: Pulling data back to host for demo purposes:
  //       In real apps, avoid synchronous reads in hot paths.
  auto Y_flat = mlx::core::reshape(Y, mlx::core::Shape{B * D});
  std::cout << "Y[0, :]: ";
  for (int j = 0; j < D; ++j) {
    // Extract element [0, j] by taking a tiny slice
    auto idx_row0 = mlx::core::full(mlx::core::Shape{1}, 0, mlx::core::int32);
    auto idx_colj = mlx::core::full(mlx::core::Shape{1}, j, mlx::core::int32);
    auto val = mlx::core::take(mlx::core::take(Y, idx_row0, /*axis=*/0), idx_colj, /*axis=*/1);
    (void)val; // placeholder; replace with scalar extraction if your API exposes it
    std::cout << "<v" << j << ">=" << val << (j + 1 < D ? ", " : "");
  }
  std::cout << "\n";

  std::cout << "Done.\n";
  return 0;
}
