"""Create the separate RAM follow-up report from isolated-container runs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
data = json.loads((DIRECTORY / "results/memory.json").read_text())
assert len(data["runs"]) == 15, "The memory experiment is incomplete"
MIB = 1024**2


def peak(row):
    return row["memory"]["after_work"]["ru_maxrss_bytes"] / MIB


def container_peak(row):
    return row["memory"]["after_work"]["memory.peak"] / MIB


names = {"keras_predict": "Current predict()", "predict_on_batch": "predict_on_batch()",
         "direct_eager": "Direct eager", "compiled": "Compiled direct"}
inference = [row for row in data["runs"] if row["phase"] == "inference"]
timed = [row for row in data["runs"] if row["phase"] == "search_timed"]
fixed = [row for row in data["runs"] if row["phase"] == "search_fixed"]
assert len({row["root_visits_sha256"] for row in fixed}) == 1
assert len({row["tree"]["nodes"] for row in fixed}) == 1
single = next(row for row in inference if row["method"] == "compiled" and row["batch_size"] == 1)
large = next(row for row in inference if row["method"] == "compiled" and row["batch_size"] == 64)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
fig.patch.set_facecolor("#f6f8fc")
for ax in axes:
    ax.set_facecolor("#f6f8fc")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
labels = [f'{names[row["method"]]} · B={row["batch_size"]}' for row in inference]
values = [peak(row) for row in inference]
axes[0].barh(labels, values, color=["#8b97aa", "#0c897c", "#8b97aa"] + ["#315ddd"] * 6)
axes[0].invert_yaxis()
axes[0].set_xlim(0, max(values) * 1.18)
axes[0].set_xlabel("Peak process RSS · MiB")
axes[0].set_title("Inference · fresh container per variant", loc="left", fontweight="bold", pad=15)
for index, value in enumerate(values):
    axes[0].text(value + 9, index, f"{value:.0f}", va="center")

y = np.arange(3)
for offset, rows, color, label in ((-0.19, timed, "#315ddd", "30 seconds of search"),
                                  (0.19, fixed, "#0c897c", "Exactly 1000 simulations")):
    values = [peak(row) for row in rows]
    axes[1].barh(y + offset, values, height=0.35, color=color, label=label)
    for index, value in enumerate(values):
        axes[1].text(value + 15, index + offset, f"{value:.0f}", va="center")
axes[1].set_yticks(y, [names[row["method"]] for row in timed])
axes[1].invert_yaxis()
axes[1].set_xlim(0, max(peak(row) for row in timed) * 1.2)
axes[1].set_xlabel("Peak process RSS · MiB")
axes[1].set_title("MCTS · same time versus same amount of work", loc="left", fontweight="bold", pad=15)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
fig.suptitle("RAM usage on one CPU", x=0.035, ha="left", fontsize=20, fontweight="bold")
fig.text(0.035, 0.90, "Same Docker image and 25×25 model · CPU quota 1 + affinity CPU 0 · no GPU · process peak includes TensorFlow", color="#526078")
fig.text(0.035, 0.025, "RSS includes resident shared library pages. Faster timed search builds a larger tree; equal-count runs retain the same tree.", color="#526078", fontsize=9)
fig.tight_layout(rect=[0.02, 0.09, 0.99, 0.88], w_pad=3)
fig.savefig(DIRECTORY / "memory.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

lines = [
    "# RAM follow-up: single-CPU Docker benchmark", "",
    f'Single-position inference peaked at {min(peak(row) for row in inference if row["batch_size"] == 1):.0f}–{max(peak(row) for row in inference if row["batch_size"] == 1):.0f} MiB of process RSS. Compiled batch 64 increased the peak by {peak(large) - peak(single):.1f} MiB relative to compiled batch 1.', "",
    "At a fixed search duration, faster inference builds more MCTS nodes and therefore uses more RAM. With a fixed 1000-simulation budget, all three evaluators produced the same root visit distribution and number of tree nodes; compare those rows to separate search size from evaluator overhead.", "",
    "## Method", "",
    "- Same Docker image, model, one-CPU quota, CPU 0 affinity, 4 GiB memory cap, no swap allowance, and no GPU as the throughput experiment.",
    "- Every case starts a fresh container and Python process; TensorFlow/Python allocator caches from another variant cannot carry over.",
    "- Inference: three warm-up calls, then five seconds of repeated calls on the same seeded 64-position corpus. Memory peaks include import, model loading, warm-up, and measurement.",
    "- Search: three warm-up calls, then either 30 seconds or exactly 1000 simulations from an empty 25×25 board. These are separate fresh processes, not consecutive searches sharing an allocator.",
    "- One memory run per case. These RAM measurements are a separate follow-up; they do not replace the earlier repeated throughput measurements.",
    "- Primary metric: process peak resident memory, from Linux `getrusage().ru_maxrss`; current RSS and anonymous/file-backed RSS are also saved from `/proc/self/status`.",
    "- Container `memory.current`, `memory.peak`, and anonymous/file/kernel breakdowns are saved separately. Cgroup accounting and process RSS have different scopes: shared library pages can appear in RSS while being charged to another cgroup. Therefore a container peak can be smaller than this process's RSS.",
    "- Values use MiB (1 MiB = 1,048,576 bytes). The 4 GiB cap is a limit, not measured consumption.", "",
    "## Inference", "",
    "| Method | Positions/call | Warmed RSS, MiB | RSS after work, MiB | Process peak, MiB | Container peak, MiB |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in inference:
    lines.append(f'| {names[row["method"]]} | {row["batch_size"]} | {row["memory"]["warmed"]["VmRSS"] / MIB:.1f} | {row["memory"]["after_work"]["VmRSS"] / MIB:.1f} | {peak(row):.1f} | {container_peak(row):.1f} |')
for heading, rows in (("MCTS: 30 seconds per method", timed), ("MCTS: exactly 1000 simulations per method", fixed)):
    lines += ["", "## " + heading, "",
              "| Method | Simulations | Seconds | Tree nodes | Materialized game states | Process peak, MiB | Container peak, MiB | RSS increase during search, MiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        memory = row["memory"]
        growth = (memory["after_work"]["VmRSS"] - memory["warmed"]["VmRSS"]) / MIB
        lines.append(f'| {names[row["method"]]} | {row["simulations"]:,} | {row["elapsed_seconds"]:.2f} | {row["tree"]["nodes"]:,} | {row["tree"]["materialized_states"]:,} | {peak(row):.1f} | {container_peak(row):.1f} | {growth:.1f} |')

lines += ["", "## Interpretation", "",
          "The main memory tradeoff is tree size under a time budget. Each expanded neural node creates a lightweight child for every legal action. Game states are materialized lazily, but the many child-node Python objects still occupy memory.", "",
          "The same-time memory increase must not be interpreted as the compiled evaluator needing that much additional RAM for one inference. It performs more searches and retains a much larger tree.", "",
          "The RSS increase during search includes allocator behavior and inference buffers as well as tree objects; it is not an exact standalone byte count of the tree. Garbage collection does not guarantee that Python or TensorFlow returns all freed pages to the operating system. Before/after snapshots are preserved in the raw results.", "",
          "These runs cover one search from one starting position. They do not establish the maximum RAM usage of a complete long game, multiple concurrent matches, other board dimensions, or different models. A single Docker CPU on an M4 Pro is not an emulation of a particular older processor.", "",
          "![RAM comparison](memory.png)", "",
          "## Artifacts", "",
          "- [Raw per-stage memory measurements and tree counts](results/memory.json)",
          "- [Isolated-container memory measurement script](memory_benchmark.py)",
          "- [Original throughput report](REPORT.md)", "",
          "Metric definitions: [Linux process memory fields](https://docs.kernel.org/filesystems/proc.html) and [cgroup v2 memory accounting](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).", ""]
(DIRECTORY / "MEMORY.md").write_text("\n".join(lines))
print("Created MEMORY.md and memory.png")
