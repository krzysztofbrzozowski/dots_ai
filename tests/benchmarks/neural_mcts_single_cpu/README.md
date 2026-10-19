# Single-CPU neural MCTS benchmark

This directory contains measurement tools and results, not production search
optimizations. See [the report](REPORT.md) after a completed run.

Run from the repository root with Docker Desktop running. The existing
`ml/models/25x25_088622_new_data_dual_head_v1.keras` checkpoint must be available.
The image uses native Linux ARM64 to match the Apple silicon host.

```bash
docker build --platform linux/arm64 \
  -t dots-mcts-single-cpu:20261019 tests/benchmarks/neural_mcts_single_cpu

mkdir -p tests/benchmarks/neural_mcts_single_cpu/results

docker run --rm --platform linux/arm64 \
  --cpus=1 --cpuset-cpus=0 --memory=4g --memory-swap=4g \
  --pids-limit=256 --network none --read-only --tmpfs /tmp:size=256m \
  --mount "type=bind,source=$(pwd),target=/workspace,readonly" \
  --mount "type=bind,source=$(pwd)/tests/benchmarks/neural_mcts_single_cpu/results,target=/results" \
  dots-mcts-single-cpu:20261019 --seconds 10 --repeats 3
```

The harness verifies both the CPU quota and single-CPU affinity, warms the
model, checks numerical agreement, and checkpoints results after every
measurement. The nine inference variants each receive three 10-second
windows. Four sequential MCTS variants each receive 10 seconds on three fixed
positions. Setup and warm-up are excluded from the measurements.

To render the completed results using the project's local environment:

```bash
MPLCONFIGDIR=/tmp/dots-benchmark-plots \
  .venv/bin/python tests/benchmarks/neural_mcts_single_cpu/render_report.py
```

The checkpoint and source checksums and complete installed package versions
are saved in `results/results.json`. The initial image was built from
`python:3.12-slim-bookworm` resolved to
`sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
Its image ID is
`sha256:31f433a27866bb8915f7a334194e758a72e61283b98c588c9f97967fa70a79fc`.
Core dependencies are pinned; the raw package list records transitive versions
for reproducing the original environment.

## RAM follow-up

The [separate memory report](MEMORY.md) distinguishes inference memory from
the growing MCTS tree. Each case uses a fresh container, measuring inference,
30 seconds of search, or exactly 1000 simulations. To repeat the measurement
and render its results from the repository root:

```bash
.venv/bin/python tests/benchmarks/neural_mcts_single_cpu/memory_benchmark.py
MPLCONFIGDIR=/tmp/dots-benchmark-plots \
  .venv/bin/python tests/benchmarks/neural_mcts_single_cpu/render_memory_report.py
```
