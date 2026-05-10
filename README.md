# Metal-Sci

A 10-task scientific-compute benchmark for **Apple Silicon Metal** kernels,
paired with a lightweight evolutionary harness for LLM-driven kernel search.

Each task ships a Metal seed kernel, a CPU reference, a roofline-anchored
fitness function over **three in-distribution problem sizes**, and **one
held-out size** the agent never sees during search. The held-out gate
$\Phi_\mathcal{T}$ is the central methodological primitive: a single
auxiliary configuration per task, evaluated once at end-of-run, that
catches confidently-wrong agent code (silent correctness violations and
silent regressions) which the in-distribution score $S_\mathcal{T}$ alone
licenses.

This repo accompanies the paper *Metal-Sci: A Scientific Compute Benchmark
for Evolutionary LLM Kernel Search on Apple Silicon*.

## What's here

- **Harness** (`metal_kernels/harness.py`): runtime-compiles `.metal`
  source via `MTLDevice.newLibraryWithSource` (no offline `xcrun metal`
  toolchain), dispatches with `MTLCommandBuffer` GPU timestamps
  (3 warmup, 10 timed, median reported), reads back through
  unified-memory `MTLBuffer.contents()`. Compile errors are returned
  as structured strings to the LLM.
- **Hardware** (`metal_kernels/hardware.py`): detects the chip
  (M1/M2/M3/M4 family) from `sysctl` and looks up peak FP32 GFLOPS +
  DRAM bandwidth for the per-size roofline ceiling.
- **Task abstraction** (`metal_kernels/task.py`): each task owns input
  generation, dispatch, CPU reference, tolerance, in-distribution sizes
  $\Sigma_\mathcal{T}$, held-out size $\sigma^\star_\mathcal{T}$, and a
  per-size roofline. The in-distribution score $S_\mathcal{T}$ is the
  geometric mean of `achieved / ceiling` across $\Sigma_\mathcal{T}$,
  hard-gated on correctness (any tolerance failure forces score $=0$).
- **Tasks** (six optimization regimes, R1–R6, plus a smoke test):

  | Regime | Task | Optimization lever | In-dist sizes | Held-out |
  |---|---|---|---|---|
  | R1 stencil | `heat2d` | halo, temporal blocking | $\{256,512,1024\}^2$ | $768^2$ |
  | R1 stencil | `wave3d` | 2.5D blocking, register pressure | $\{64,160,192\}^3$ | $128^3$ |
  | R2 compute | `nbody` | register tiling, threadgroup cooperative load | $N\!\in\!\{256,1024,2048\}$ | $512$ |
  | R2 compute | `hmc` | per-thread state vs. register file | $(d,K)\!\in\!\{(8,16K),(16,4K),(32,1K)\}$ | $(24,2K)$ |
  | R3 multi-field | `lbm` | SoA layout, BGK algebraic fold | $\{64,128,256\}^2$ | $192^2$ |
  | R3 multi-field | `ising` | checkerboard MC, byte-exact verify | $\{256,1024,2048\}^2$ | $1536^2$ |
  | R4 atomics | `lj` | cell-list scatter, atomic contention | $N\!\in\!\{1.7,4.1,10.6\}\mathrm{K}$ | $2744$ |
  | R5 multi-kernel | `gradshaf` | in-kernel reduction + var-coef stencil | $\{65,257,513\}^2$ | $129^2$ |
  | R6 butterfly | `fft3d` | TG bank conflicts, mixed-radix, `simd_shuffle` | $\{32,64,128\}^3$ | $256^3$ |
  | (smoke) | `saxpy` | DRAM saturation | $\{1,16,64\}\mathrm{M}$ | $4\mathrm{M}$ |

- **LLM bridge** (`metal_kernels/llm.py`): single `call_llm` entry that
  dispatches to Claude (via `claude_agent_sdk`), Gemini (via
  `google-genai`), or OpenAI (via the `openai` SDK, including reasoning
  models like `gpt-5.5`).
- **Evolution loop** (`metal_kernels/evolve.py`): seed → iterate; each
  iteration sees the previous candidate, the incumbent best, and a
  short history. Strict $(1{+}1)$ promotion (replace incumbent only on
  strict $S_\mathcal{T}$ improvement). Persists prompts, responses,
  sources, and JSON results.

## Quickstart

Verify the seed kernels compile, pass correctness, and time:

```sh
uv run run_benchmark.py --task saxpy    --evaluate-seed-only
uv run run_benchmark.py --task heat2d   --evaluate-seed-only
uv run run_benchmark.py --task wave3d   --evaluate-seed-only
uv run run_benchmark.py --task nbody    --evaluate-seed-only
uv run run_benchmark.py --task hmc      --evaluate-seed-only
uv run run_benchmark.py --task lbm      --evaluate-seed-only
uv run run_benchmark.py --task ising    --evaluate-seed-only
uv run run_benchmark.py --task lj       --evaluate-seed-only
uv run run_benchmark.py --task gradshaf --evaluate-seed-only
uv run run_benchmark.py --task fft3d    --evaluate-seed-only
```

Run an evolution loop with Claude, Gemini, or GPT:

```sh
# Claude via the Agent SDK (requires ANTHROPIC_API_KEY)
uv run run_benchmark.py --task hmc      --model claude-opus-4-7        --iterations 10

# Gemini (requires GEMINI_API_KEY or GOOGLE_API_KEY)
uv run run_benchmark.py --task gradshaf --model gemini-3.1-pro-preview --iterations 10

# OpenAI reasoning model (requires OPENAI_API_KEY)
uv run run_benchmark.py --task fft3d    --model gpt-5.5                --iterations 10
```

Per run, an output directory is created under `results/` containing:

```
00_seed.metal       # the unchanged seed
01_prompt.md        # the user prompt sent to the LLM at iteration 1
01_response.md      # raw LLM response
01_reasoning.md     # extended-thinking tokens (when available)
01_candidate.metal  # extracted Metal source
01_result.json      # per-size correctness + timing + fraction-of-ceiling
...
best.metal          # incumbent at end of run
best_result.json
history.json        # per-iteration record
summary.json
```

The held-out evaluation $\Phi_\mathcal{T}$ is computed by separate scripts
(see `results/_run_logs/eval_held_out*.py`) on the run's incumbent at
$\sigma^\star_\mathcal{T}$, and is **never** included in the feedback
packet $\mathcal{F}_k$ the LLM sees during search.

## Reference results (Apple M1 Pro, 4500 GFLOPS / 200 GB/s)

Three matched single-model sweeps over the 10 tasks at the same per-task
iteration budget (10 each except `lbm` at 25 and `wave3d` at 15),
$\mu{=}1{+}\lambda{=}1$, no human prompt intervention.
*In-dist. ×* = best/seed, gmean over the three in-distribution sizes.
*Held-out ×* = best/seed at the unseen size; **bold** marks meaningful
improvements ($\geq 1.05\times$).

|  | In-dist. × |  |  | Held-out × |  |  |  |
|---|---|---|---|---|---|---|---|
| Task | Opus | Gemini | GPT | Opus | Gemini | GPT | Outcome |
| `saxpy`    | **1.25** | 1.00     | 1.01     | **1.17** | 0.98     | 0.98     | saturated |
| `heat2d`   | 1.00     | 1.03     | 1.00     | 0.86     | 1.01     | 0.82     | saturated |
| `wave3d`   | **1.26** | 1.00     | 1.00     | 1.00     | 0.90     | 0.99     | saturated |
| `ising`    | **1.13** | 1.00     | **1.09** | 0.94     | 0.99     | 0.88     | flat |
| `fft3d`    | 1.03     | **1.19** | **2.95** | **1.12** | **1.20** | **0.23** | **GPT silent regression** |
| `nbody`    | **2.83** | **2.00** | **2.19** | **1.24** | **1.50** | **1.37** | generalizes |
| `gradshaf` | **1.89** | **2.89** | **1.93** | **2.05** | **2.91** | **1.86** | generalizes |
| `lj`       | **1.77** | **1.98** | **1.62** | **1.24** | **1.87** | **1.34** | generalizes |
| `lbm`      | **1.46** | **1.06** | **1.33** | 0.97     | **1.16** | 1.01     | tied at $192^2$ |
| `hmc`      | **10.6** | **10.7** | **7.19** | **FAIL** | **17.6** | **18.6** | **Opus wrong at $d{=}24$**; Gemini, GPT generalize |

(Opus = `claude-opus-4-7`, Gemini = `gemini-3.1-pro-preview`,
GPT = `gpt-5.5`.)

The two diagnostic cells are the central evidence for $\Phi_\mathcal{T}$
as an oversight primitive:

- **Silent correctness violation (`hmc`, Opus).** The incumbent dispatches
  `if (d==8) run<8>() ... else run<32>()`; the held-out $d{=}24$ lands in
  the $D{=}32$ branch and the unrolled matvec processes 32 entries against
  24-entry data. In-distribution looks $10.6\times$ faster; the
  sample covariance is $\sim\!10\sigma$ off target.
- **Silent performance regression (`fft3d`, GPT).** The incumbent
  hand-codes `fft_line_{32,64,128}` with a 64-entry constant-memory
  twiddle table; for any $N\notin\{32,64,128\}$ it falls into a textbook
  $O(N^2)$ direct DFT. At held-out $N{=}256$ this costs $\sim\!32\times$
  more arithmetic per output than the seed's $O(N\log N)$ Stockham FFT,
  flipping a reported $2.95\times$ in-dist. win into a $0.23\times$
  deployment-grade slowdown.

A rough generation-time profile at high reasoning budgets: Opus
$\sim 0.6$ min/iter, Gemini $\sim 3.5$ min/iter, GPT $\sim 6.6$ min/iter.

## Adding a new task

Drop a `seeds/<name>.metal` and a `metal_kernels/tasks/<name>.py`
implementing `Task.evaluate_size`, declaring `in_distribution_sizes` and
`held_out_sizes`, and providing a CPU reference + tolerance. Decorate
with `@register_task("<name>")` and add the import to
`metal_kernels/tasks/__init__.py`. The harness plumbing (compile,
multi-size loop, scoring, correctness gate, $\Phi_\mathcal{T}$) is shared.
