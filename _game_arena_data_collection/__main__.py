"""Run with python -m _game_arena_data_collection --minutes 10."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .audit import audit_collection
from .collector import collect
from .config import COLLECTION_CONFIG, DEFAULT_OUTPUT_DIRECTORY, DEFAULT_WORKERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=10, help="Finish active games after this time (default: 10).")
    parser.add_argument("--games", type=int, help="Maximum additional games in this invocation.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent games in separate CPU processes (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--seed", type=int, default=COLLECTION_CONFIG.seed)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--audit-only", action="store_true", help="Replay and inspect the existing collection without adding games.")
    args = parser.parse_args()
    config = replace(COLLECTION_CONFIG, seed=args.seed)
    directory = args.output_directory / f"{config.rows}x{config.cols}"
    if not args.audit_only:
        session = collect(
            config, output_directory=args.output_directory,
            duration_seconds=args.minutes * 60, max_games=args.games,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
        print(json.dumps(session, indent=2))
    summary = audit_collection(directory)
    print(json.dumps(summary, indent=2))
    print(f"Report: {directory / 'report.md'}")


if __name__ == "__main__":
    main()
