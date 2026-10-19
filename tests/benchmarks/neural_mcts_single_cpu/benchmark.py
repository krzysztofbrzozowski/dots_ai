"""Read-only model/search experiments; does not alter production inference."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
import keras


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from game.enclosure import DotsGame
from mcts.neural_search import NeuralMCTSNode, NeuralMonteCarloTreeSearch
from ml.predictor import DualHeadPredictor, game_state_to_model_input


def read_optional(path):
    path = Path(path)
    return path.read_text().strip() if path.exists() else None


def materialize(outputs):
    return tf.nest.map_structure(
        lambda value: value.numpy() if hasattr(value, "numpy") else value,
        outputs,
    )


def make_positions(seed):
    rng = np.random.default_rng(seed)
    state = DotsGame(25, 25)
    snapshots = {0: state}
    for move in range(1, 190):
        legal = state.get_legal_actions()
        if not legal:
            raise RuntimeError("The deterministic benchmark game ended too early")
        state = state.move(legal[int(rng.integers(len(legal)))])
        snapshots[move] = state
    return [snapshots[index * 3] for index in range(64)], {
        "opening": snapshots[0],
        "after_60_moves": snapshots[60],
        "after_150_moves": snapshots[150],
    }


def time_inference(function, batches, seconds, batch_size):
    durations = []
    call_count = 0
    cpu_started = time.process_time()
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        before = time.perf_counter()
        materialize(function(batches[call_count % len(batches)]))
        durations.append(time.perf_counter() - before)
        call_count += 1
    elapsed = time.perf_counter() - started
    return {
        "calls": call_count,
        "positions": call_count * batch_size,
        "elapsed_seconds": elapsed,
        "cpu_seconds": time.process_time() - cpu_started,
        "positions_per_second": call_count * batch_size / elapsed,
        "p50_call_ms": statistics.median(durations) * 1000,
        "p95_call_ms": float(np.percentile(durations, 95)) * 1000,
    }


def summarize(records):
    grouped = {}
    for record in records:
        grouped.setdefault((record["method"], record["batch_size"]), []).append(record)
    summary = []
    for (method, batch_size), rows in grouped.items():
        positions = sum(row["positions"] for row in rows)
        elapsed = sum(row["elapsed_seconds"] for row in rows)
        rates = [row["positions_per_second"] for row in rows]
        summary.append({
            "method": method,
            "positions_per_call": batch_size,
            "repeats": len(rows),
            "total_positions": positions,
            "total_calls": sum(row["calls"] for row in rows),
            "total_elapsed_seconds": elapsed,
            "positions_per_second": positions / elapsed,
            "min_repeat_positions_per_second": min(rates),
            "max_repeat_positions_per_second": max(rates),
            "mean_p50_call_ms": statistics.mean(row["p50_call_ms"] for row in rows),
            "mean_p95_call_ms": statistics.mean(row["p95_call_ms"] for row in rows),
            "cpu_seconds_per_wall_second": sum(row["cpu_seconds"] for row in rows) / elapsed,
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/results/results.json"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")

    affinity = sorted(os.sched_getaffinity(0))
    cpu_max = read_optional("/sys/fs/cgroup/cpu.max")
    if len(affinity) != 1:
        raise RuntimeError(f"Expected one eligible CPU; affinity is {affinity}")
    if cpu_max is None or cpu_max.split()[0] == "max":
        raise RuntimeError("Expected an explicit cgroup-v2 CPU quota")
    quota, period = map(int, cpu_max.split())
    if quota / period != 1:
        raise RuntimeError(f"Expected a one-CPU quota, got {cpu_max}")

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("This benchmark must run without a GPU")

    checkpoint = ROOT / "ml/models/25x25_088622_new_data_dual_head_v1.keras"
    predictor = DualHeadPredictor(checkpoint)
    model = predictor._load_model()
    corpus, search_positions = make_positions(20261019)
    encoded = [game_state_to_model_input(state) for state in corpus]
    all_inputs = tuple(np.concatenate([pair[index] for pair in encoded]) for index in (0, 1))

    @tf.function(input_signature=[
        tf.TensorSpec([None, 25, 25, 5], tf.float32),
        tf.TensorSpec([None, 2], tf.float32),
    ], autograph=False)
    def compiled(board, scores):
        return model((board, scores), training=False)

    functions = {
        "keras_predict": lambda inputs: model.predict(inputs, verbose=0),
        "predict_on_batch": model.predict_on_batch,
        "direct_eager": lambda inputs: model(inputs, training=False),
        "compiled": lambda inputs: compiled(*inputs),
    }
    variants = [
        ("keras_predict", 1),
        ("predict_on_batch", 1),
        ("direct_eager", 1),
        *(("compiled", size) for size in (1, 4, 8, 16, 32, 64)),
    ]
    batches = {
        size: [tuple(value[start:start + size] for value in all_inputs)
               for start in range(0, len(corpus), size)]
        for size in (1, 4, 8, 16, 32, 64)
    }

    payload = {
        "metadata": {
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in tf.config.list_physical_devices()],
            "affinity": affinity,
            "cpu_max": cpu_max,
            "memory_max": read_optional("/sys/fs/cgroup/memory.max"),
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_parameters": model.count_params(),
            "corpus_size": len(corpus),
            "corpus_sha256": hashlib.sha256(b"".join(value.tobytes() for value in all_inputs)).hexdigest(),
            "seconds_per_measurement": args.seconds,
            "inference_repeats": args.repeats,
            "warmup_calls_per_variant": 3,
            "seed": 20261019,
            "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout.splitlines(),
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("ml/predictor.py", "mcts/neural_search.py", "game/engine.py", "tests/benchmarks/neural_mcts_single_cpu/benchmark.py")
            },
        },
        "correctness": [],
        "inference_runs": [],
        "search_runs": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload["inference_summary"] = summarize(payload["inference_runs"])
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    reference = model.predict(all_inputs, batch_size=1, verbose=0)
    for method, size in variants:
        function = functions[method]
        for index in range(3):
            materialize(function(batches[size][index % len(batches[size])]))
        output = materialize(function(batches[size][0]))
        differences = {}
        for head in ("policy", "value"):
            expected = reference[head][:size]
            np.testing.assert_allclose(output[head], expected, rtol=1e-4, atol=1e-5)
            differences[head] = float(np.max(np.abs(output[head] - expected)))
        payload["correctness"].append({"method": method, "batch_size": size, "max_abs_difference": differences})
    save()
    print("SETUP " + json.dumps({key: payload["metadata"][key] for key in ("machine", "affinity", "cpu_max", "model_parameters", "corpus_size")}), flush=True)
    print("All inference variants passed output-equivalence checks.", flush=True)

    order_rng = random.Random(20261019)
    for repeat in range(args.repeats):
        order = list(variants)
        order_rng.shuffle(order)
        for method, size in order:
            gc.collect()
            record = time_inference(functions[method], batches[size], args.seconds, size)
            record.update(method=method, batch_size=size, repeat=repeat + 1)
            payload["inference_runs"].append(record)
            save()
            print("INFERENCE " + json.dumps(record), flush=True)

    class ExperimentalModelAdapter:
        input_shape = model.input_shape
        output_names = model.output_names

        def __init__(self, function):
            self.function = function

        def predict(self, inputs, verbose=0):
            return materialize(self.function(inputs))

    evaluators = {"current_predictor": predictor.predict_state_policy_value}
    for name in ("direct_eager", "predict_on_batch", "compiled"):
        experimental_predictor = DualHeadPredictor(checkpoint)
        experimental_predictor._model = ExperimentalModelAdapter(functions[name])
        evaluators[name] = experimental_predictor.predict_state_policy_value

    # A fixed simulation count checks that adapters preserve the actual search,
    # independently of the different exploration counts in timed runs.
    expected_visits = None
    for name, evaluator in evaluators.items():
        root = NeuralMCTSNode(state=search_positions["opening"])
        search = NeuralMonteCarloTreeSearch(root, evaluator)
        selected = search.best_action(simulations_number=32)
        visits = [(node.action, node.n) for node in root.children]
        if expected_visits is None:
            expected_visits = visits
        if visits != expected_visits:
            raise RuntimeError(f"Search visit counts changed with {name}")
        payload["correctness"].append({"search_method": name, "fixed_simulations": 32, "identical_root_visits": True, "selected_action": selected.action})
    save()

    for position_name, state in search_positions.items():
        order = list(evaluators)
        order_rng.shuffle(order)
        for method in order:
            evaluator = evaluators[method]
            evaluator(state)
            gc.collect()
            timings = {"calls": 0, "seconds": 0.0}

            def timed_evaluator(position):
                started = time.perf_counter()
                result = evaluator(position)
                timings["seconds"] += time.perf_counter() - started
                timings["calls"] += 1
                return result

            root = NeuralMCTSNode(state=state)
            search = NeuralMonteCarloTreeSearch(root, timed_evaluator)
            cpu_started = time.process_time()
            selected = search.best_action(total_simulation_seconds=args.seconds)
            elapsed = search.last_search_stats.elapsed_seconds
            record = {
                "method": method,
                "position": position_name,
                "requested_seconds": args.seconds,
                "elapsed_seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu_started,
                "simulations": search.last_search_stats.completed_rollouts,
                "simulations_per_second": search.last_search_stats.completed_rollouts / elapsed,
                "inference_calls": timings["calls"],
                "inference_seconds": timings["seconds"],
                "inference_fraction": timings["seconds"] / elapsed,
                "selected_action": selected.action,
                "root_children": len(root.children),
            }
            payload["search_runs"].append(record)
            save()
            print("SEARCH " + json.dumps(record), flush=True)
            del search, root
    payload["metadata"]["cpu_stat_at_end"] = read_optional("/sys/fs/cgroup/cpu.stat")
    save()
    print("COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
