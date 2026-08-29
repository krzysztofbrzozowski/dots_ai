"""Run the saved MCTS game analyzer independently from ``main_mcts.py``."""

import uvicorn

from analysis.server import app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001


def main():
    uvicorn.run(
        app,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        access_log=False,
    )


if __name__ == "__main__":
    main()

