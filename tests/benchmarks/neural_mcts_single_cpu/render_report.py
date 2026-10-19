"""Render the completed benchmark's measurements as a report and chart."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
data = json.loads((DIRECTORY / "results/results.json").read_text())
assert len(data["inference_runs"]) == 27, "The inference experiment is incomplete"
assert len(data["search_runs"]) == 12, "The search experiment is incomplete"

order = [
    ("keras_predict", 1), ("predict_on_batch", 1), ("direct_eager", 1),
    *(("compiled", size) for size in (1, 4, 8, 16, 32, 64)),
]
labels = {
    "keras_predict": "Keras predict()",
    "predict_on_batch": "predict_on_batch()",
    "direct_eager": "Direct eager",
    "compiled": "Compiled direct",
    "current_predictor": "Current predictor",
}
lookup = {(row["method"], row["positions_per_call"]): row for row in data["inference_summary"]}
rows = [lookup[key] for key in order]
baseline = rows[0]["positions_per_second"]
search_order = ["current_predictor", "direct_eager", "predict_on_batch", "compiled"]
search_summary = []
for method in search_order:
    records = [row for row in data["search_runs"] if row["method"] == method]
    elapsed = sum(row["elapsed_seconds"] for row in records)
    simulations = sum(row["simulations"] for row in records)
    search_summary.append({"method": method, "simulations": simulations, "elapsed_seconds": elapsed,
                           "simulations_per_second": simulations / elapsed})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor("#f6f8fc")
for ax in axes:
    ax.set_facecolor("#f6f8fc")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)

names = [f'{labels[row["method"]]} · B={row["positions_per_call"]}' for row in rows]
rates = [row["positions_per_second"] for row in rows]
low = [row["positions_per_second"] - row["min_repeat_positions_per_second"] for row in rows]
high = [row["max_repeat_positions_per_second"] - row["positions_per_second"] for row in rows]
colors = ["#8b97aa", "#0c897c", "#8b97aa"] + ["#315ddd"] * 6
axes[0].barh(names, rates, color=colors, xerr=np.array([low, high]), capsize=3, error_kw={"linewidth": 0.8})
axes[0].invert_yaxis()
axes[0].set_xlim(0, max(rates) * 1.22)
axes[0].set_xlabel("Model positions / second")
axes[0].set_title("Inference only · 3 × 10 s per variant", loc="left", fontweight="bold", pad=15)
for index, rate in enumerate(rates):
    axes[0].text(rate + max(rates) * 0.035, index, f"{rate:.1f}", va="center", fontsize=10)

positions = ["opening", "after_60_moves", "after_150_moves"]
position_labels = ["Opening", "After 60 moves", "After 150 moves"]
y = np.arange(len(search_order))
width = 0.23
for index, (position, color) in enumerate(zip(positions, ["#315ddd", "#0c897c", "#d69b35"])):
    rates_for_position = [next(row["simulations_per_second"] for row in data["search_runs"]
                               if row["method"] == method and row["position"] == position)
                          for method in search_order]
    axes[1].barh(y + (index - 1) * width, rates_for_position, height=width, color=color,
                 label=position_labels[index])
axes[1].set_yticks(y, [labels[method] for method in search_order])
axes[1].invert_yaxis()
axes[1].set_xlabel("Complete MCTS simulations / second")
axes[1].set_title("Actual sequential MCTS · 10 s per position", loc="left", fontweight="bold", pad=15)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=3, frameon=False, fontsize=8.5)

fig.suptitle("Neural MCTS on one CPU", x=0.035, ha="left", fontsize=20, fontweight="bold")
fig.text(0.035, 0.90, "Docker Linux ARM64 · Apple M4 Pro host · CPU quota 1 + affinity CPU 0 · no GPU · 25×25 checkpoint", color="#526078")
fig.text(0.035, 0.025, "Inference error bars show the range of three runs. Batched inference throughput is not batched-MCTS throughput.", color="#526078", fontsize=9)
fig.tight_layout(rect=[0.02, 0.09, 0.99, 0.88], w_pad=3)
fig.savefig(DIRECTORY / "comparison.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

metadata = data["metadata"]
compiled_single = lookup[("compiled", 1)]
best_inference = max(rows, key=lambda row: row["positions_per_second"])
best_search = max(search_summary, key=lambda row: row["simulations_per_second"])
lines = [
    "# Neural MCTS: single-CPU Docker benchmark", "",
    "Measured on 19 October 2026. Production game, MCTS, and prediction code were not changed.", "",
    f'Compiled single-position inference was {compiled_single["positions_per_second"] / baseline:.2f}× faster than the current `predict()` API. The fastest measured inference configuration, {labels[best_inference["method"]]} with batch {best_inference["positions_per_call"]}, improved throughput by a further {(best_inference["positions_per_second"] / compiled_single["positions_per_second"] - 1) * 100:.1f}% over compiled batch 1.', "",
    f'For actual sequential MCTS, the fastest measured evaluator was {labels[best_search["method"]]}, at {best_search["simulations_per_second"] / search_summary[0]["simulations_per_second"]:.2f}× the baseline throughput across the three tested positions.', "",
    "For this single-CPU setup, start with low-overhead inference at batch size 1 (`predict_on_batch()` or a compiled direct call). Larger batches offer a smaller additional model-throughput gain and delay tree feedback; their actual MCTS benefit remains unmeasured.", "",
    "## Environment and method", "",
    "- Host: Apple M4 Pro; native Linux ARM64 container, without x86 emulation or a GPU.",
    "- Docker limits: `--cpus=1 --cpuset-cpus=0 --memory=4g --memory-swap=4g`.",
    f'- Verified from inside the container: affinity `{metadata["affinity"]}`, cgroup `cpu.max = {metadata["cpu_max"]}`, memory limit `{metadata["memory_max"]}` bytes.',
    "- TensorFlow intra-op/inter-op threads, OpenMP, OpenBLAS, and MKL were limited to one.",
    f'- TensorFlow {metadata["tensorflow"]}, Keras {metadata["keras"]}, NumPy {metadata["numpy"]}. The full package list is saved in the raw results.',
    f'- Checkpoint: `{metadata["checkpoint"]}`; {metadata["model_parameters"]:,} parameters; unchanged float32 weights.',
    f'- Model SHA-256: `{metadata["checkpoint_sha256"]}`.',
    "- A seeded corpus contains 64 distinct legal game states, every third move from move 0 through 189, encoded as board and score inputs.",
    "- Every inference variant receives three 10-second measurement windows, with randomized order in each round. Model loading, graph tracing, and warm-up are excluded.",
    "- Each timed call includes conversion of both output heads to NumPy, so asynchronous dispatch alone is not counted as completed inference.",
    "- Inference inputs are prepared before timing. Complete search timings additionally include state preparation, prediction postprocessing, PUCT selection, expansion, and backpropagation.",
    "- A call/search simulation already started at the deadline is allowed to finish. Rates use actual elapsed time, including that overshoot.",
    "- The repository is mounted read-only. The container has no network during measurement; only the results directory is writable.", "",
    "## Model inference", "",
    "Totals below are actual completed positions across three windows (approximately 30 seconds). Throughput uses exact elapsed seconds.", "",
    "| Method | Positions/call | Positions/s | Range across runs | Mean per-run p50 call, ms | Total positions | Actual seconds | Speedup vs predict |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(f'| {labels[row["method"]]} | {row["positions_per_call"]} | {row["positions_per_second"]:.1f} | {row["min_repeat_positions_per_second"]:.1f}–{row["max_repeat_positions_per_second"]:.1f} | {row["mean_p50_call_ms"]:.2f} | {row["total_positions"]:,} | {row["total_elapsed_seconds"]:.2f} | {row["positions_per_second"] / baseline:.2f}× |')

lines += ["", "`Compiled direct` means `model(inputs, training=False)` inside a warmed `tf.function`, with a dynamic batch dimension. XLA, quantization, reduced precision, model compression, and retraining are not used.", "",
          "`predict_on_batch()` is measured at batch size 1 here: its gain is from the inference API, not from evaluating multiple positions at once.", "",
          "## Complete sequential MCTS", "",
          "Each method receives 10 seconds on the same three initial positions: empty board, after 60 moves, and after 150 moves. All methods still evaluate one leaf per call. Experimental in-memory adapters reuse the existing predictor's preprocessing and policy/value conversion.", "",
          "| Method | Opening sims/s | After 60 moves sims/s | After 150 moves sims/s | Total simulations | Actual seconds | Pooled sims/s | Gain |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
search_base = search_summary[0]["simulations_per_second"]
for row in search_summary:
    values = [next(record["simulations_per_second"] for record in data["search_runs"] if record["method"] == row["method"] and record["position"] == position) for position in positions]
    lines.append(f'| {labels[row["method"]]} | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} | {row["simulations"]:,} | {row["elapsed_seconds"]:.2f} | {row["simulations_per_second"]:.1f} | {row["simulations_per_second"] / search_base:.2f}× |')

max_diff = max(max(record["max_abs_difference"].values()) for record in data["correctness"] if "max_abs_difference" in record)
lines += ["", "## Correctness and limits", "",
          f"All nine inference variants passed numerical comparison against the single-position baseline (`rtol=1e-4`, `atol=1e-5`); maximum observed absolute output difference: `{max_diff:.8g}`.", "",
          "All four evaluator variants also produced identical root visit counts and the same selected move in a fixed 32-simulation opening search. This is a sanity check, not a playing-strength evaluation.", "",
          "Batched rows measure inference only. A batched MCTS scheduler with path reservations has not been implemented or timed, so those rates must not be presented as actual MCTS simulations per second. Delayed feedback and pending-leaf selection can change search quality.", "",
          "A Docker single-CPU limit restricts available resources; it does not turn an M4 Pro core into an older or slower CPU. These relative results apply to this ARM64 container/runtime, not universally to every processor. Other host workloads and thermal behavior are not controlled. Three samples support a practical comparison, not a formal confidence interval.", "",
          "Keeping a 30-second move budget increases the number of simulations after optimization. Reducing the budget while retaining playing strength requires separate game comparisons.", "",
          "![Inference and complete-search comparison](comparison.png)", "",
          "## Artifacts and reproduction", "",
          "- [Raw measurements, environment, package versions, checksums, and correctness checks](results/results.json)",
          "- [Benchmark harness](benchmark.py)",
          "- [Docker image definition](Dockerfile)",
          "- [Core dependency versions](requirements.txt)",
          "- [Reproduction instructions](README.md)", "",
          "The interpretation of CPU quota and CPU affinity follows [Docker's resource constraints documentation](https://docs.docker.com/engine/containers/resource_constraints/). The alternative inference APIs are documented in [Keras model training APIs](https://keras.io/api/models/model_training_apis/) and [TensorFlow's tf.function guide](https://www.tensorflow.org/guide/function).", ""]
(DIRECTORY / "REPORT.md").write_text("\n".join(lines))
(DIRECTORY / "results/summary.json").write_text(json.dumps({"inference": rows, "search": search_summary}, indent=2) + "\n")
print("Created REPORT.md, comparison.png, and results/summary.json")
