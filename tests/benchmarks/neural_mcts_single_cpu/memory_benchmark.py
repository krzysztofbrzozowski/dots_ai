"""Measure memory in a fresh, single-CPU Docker container for each case."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
IMAGE = "dots-mcts-single-cpu:20261019"


def memory_snapshot():
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
            status[key] = int(value.split()[0]) * 1024
    status["ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.current"):
        status[name] = int((Path("/sys/fs/cgroup") / name).read_text().strip())
    statistics = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
    status["cgroup_breakdown"] = {name: int(statistics[name]) for name in ("anon", "file", "kernel")}
    return status


def worker(args):
    result = {
        "phase": args.phase,
        "method": args.method,
        "batch_size": args.batch_size,
        "requested_seconds": args.seconds if args.phase != "search_fixed" else None,
        "requested_simulations": args.simulations if args.phase == "search_fixed" else None,
        "memory": {"python_only": memory_snapshot()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    }
    if result["affinity"] != [0] or result["cpu_max"] != "100000 100000":
        raise RuntimeError("Expected CPU 0 affinity and a one-CPU quota")

    import numpy as np
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("The memory benchmark must not use a GPU")
    sys.path.insert(0, str(ROOT))
    from game.enclosure import DotsGame
    from ml.predictor import DualHeadPredictor, game_state_to_model_input
    from mcts.neural_search import NeuralMCTSNode, NeuralMonteCarloTreeSearch

    result["memory"]["libraries_loaded"] = memory_snapshot()
    checkpoint = ROOT / "ml/models/25x25_088622_new_data_dual_head_v1.keras"
    predictor = DualHeadPredictor(checkpoint)
    model = predictor._load_model()
    result["model_parameters"] = model.count_params()
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result["tensorflow"] = tf.__version__
    result["memory"]["model_loaded"] = memory_snapshot()

    def materialize(outputs):
        return tf.nest.map_structure(lambda value: value.numpy() if hasattr(value, "numpy") else value, outputs)

    @tf.function(input_signature=[tf.TensorSpec([None, 25, 25, 5], tf.float32),
                                 tf.TensorSpec([None, 2], tf.float32)], autograph=False)
    def compiled(board, scores):
        return model((board, scores), training=False)

    functions = {
        "keras_predict": lambda inputs: model.predict(inputs, verbose=0),
        "predict_on_batch": model.predict_on_batch,
        "direct_eager": lambda inputs: model(inputs, training=False),
        "compiled": lambda inputs: compiled(*inputs),
    }
    function = functions[args.method]

    if args.phase == "inference":
        from tests.benchmarks.neural_mcts_single_cpu.benchmark import make_positions
        corpus, _ = make_positions(20261019)
        encoded = [game_state_to_model_input(state) for state in corpus]
        all_inputs = tuple(np.concatenate([pair[index] for pair in encoded]) for index in (0, 1))
        batches = [tuple(value[start:start + args.batch_size] for value in all_inputs)
                   for start in range(0, len(corpus), args.batch_size)]
        for index in range(3):
            materialize(function(batches[index % len(batches)]))
        gc.collect()
        result["memory"]["warmed"] = memory_snapshot()
        started = time.perf_counter()
        calls = 0
        while time.perf_counter() - started < args.seconds:
            materialize(function(batches[calls % len(batches)]))
            calls += 1
        result["elapsed_seconds"] = time.perf_counter() - started
        result["calls"] = calls
        result["positions"] = calls * args.batch_size
        result["memory"]["after_work"] = memory_snapshot()
    else:
        state = DotsGame(25, 25)

        class Adapter:
            input_shape = model.input_shape
            output_names = model.output_names

            def predict(self, inputs, verbose=0):
                return materialize(function(inputs))

        if args.method != "keras_predict":
            predictor._model = Adapter()
        for _ in range(3):
            predictor.predict_state_policy_value(state)
        gc.collect()
        result["memory"]["warmed"] = memory_snapshot()
        root = NeuralMCTSNode(state=state)
        search = NeuralMonteCarloTreeSearch(root, predictor.predict_state_policy_value)
        selected = search.best_action(
            simulations_number=args.simulations if args.phase == "search_fixed" else None,
            total_simulation_seconds=args.seconds if args.phase == "search_timed" else None,
        )
        result["elapsed_seconds"] = search.last_search_stats.elapsed_seconds
        result["simulations"] = search.last_search_stats.completed_rollouts
        result["selected_action"] = selected.action
        result["memory"]["after_work"] = memory_snapshot()
        pending = [root]
        counts = {"nodes": 0, "expanded_nodes": 0, "materialized_states": 0}
        while pending:
            node = pending.pop()
            counts["nodes"] += 1
            counts["expanded_nodes"] += int(node.is_expanded)
            counts["materialized_states"] += int(node._state is not None)
            pending.extend(node.children)
        result["tree"] = counts
        visits = [(node.action, node.n) for node in root.children]
        result["root_visits_sha256"] = hashlib.sha256(json.dumps(visits).encode()).hexdigest()
        del selected, search, root, node, pending
    gc.collect()
    result["memory"]["after_gc"] = memory_snapshot()
    print(json.dumps(result), flush=True)


def orchestrate():
    cases = [
        ("inference", "keras_predict", 1),
        ("inference", "predict_on_batch", 1),
        ("inference", "direct_eager", 1),
        *(("inference", "compiled", size) for size in (1, 4, 8, 16, 32, 64)),
        *((phase, method, 1) for phase in ("search_timed", "search_fixed")
          for method in ("keras_predict", "predict_on_batch", "compiled")),
    ]
    output = DIRECTORY / "results/memory.json"
    payload = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "image": IMAGE, "inference_seconds": 5,
               "search_seconds": 30, "fixed_simulations": 1000,
               "note": "Each case uses a fresh container/process. One run per case; peaks include runtime initialization.",
               "runs": []}
    for index, (phase, method, batch_size) in enumerate(cases):
        command = [
            "docker", "run", "--rm", "--name", f"dots-mcts-ram-{phase}-{method}-{batch_size}",
            "--platform", "linux/arm64", "--cpus=1", "--cpuset-cpus=0",
            "--memory=4g", "--memory-swap=4g", "--pids-limit=256",
            "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m",
            "--mount", f"type=bind,source={ROOT},target=/workspace,readonly",
            "--entrypoint", "python", IMAGE,
            "/workspace/tests/benchmarks/neural_mcts_single_cpu/memory_benchmark.py",
            "--worker", "--phase", phase, "--method", method,
            "--batch-size", str(batch_size), "--seconds", "5" if phase == "inference" else "30",
            "--simulations", "1000",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(f'{phase}/{method}/{batch_size}: {completed.stdout}\n{completed.stderr}')
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["runs"].append(record)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        peak = record["memory"]["after_work"]["ru_maxrss_bytes"] / 1024**2
        cgroup_peak = record["memory"]["after_work"]["memory.peak"] / 1024**2
        print(f'{index + 1}/{len(cases)} {phase} {method} batch={batch_size}: '
              f'process peak={peak:.1f} MiB; container peak={cgroup_peak:.1f} MiB; '
              f'simulations={record.get("simulations", "n/a")}', flush=True)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", choices=("inference", "search_timed", "search_fixed"))
    parser.add_argument("--method", choices=("keras_predict", "predict_on_batch", "direct_eager", "compiled"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--simulations", type=int, default=1000)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        orchestrate()
