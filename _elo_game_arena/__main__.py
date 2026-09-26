"""Start the arena with ``python -m _elo_game_arena``."""

import uvicorn

from .config import HOST, PORT


def main():
    uvicorn.run(
        "_elo_game_arena.server:app",
        host=HOST,
        port=PORT,
        access_log=False,
    )


if __name__ == "__main__":
    main()
