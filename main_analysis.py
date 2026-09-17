"""Run saved-game analysis and disposable MCTS continuation experiments."""

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
