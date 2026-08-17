"""Backward-compatible public API for the Dots game

Implementation lives in focused modules; imports from ``enclosure`` continue
to work for existing callers
"""

try:  # Package imports using the game enclosure module
    from .board import (
        EMPTY,
        PLAYER_1,
        PLAYER_2,
        PLAYERS,
        get_neighbors,
        opponent_of,
    )
    from .capture import (
        CaptureInfo,
        _could_have_closed_loop,
        detect_capture,
        detect_capture_info,
        find_candidate_regions,
        find_enclosed_regions,
        flood_fill_region,
    )
    from .engine import DotsGame
    from .groups import UnionFind, find_group, merge_groups, rebuild_groups
    from .rendering import (
        BLOCKED_SYMBOL,
        COLORS,
        SYMBOLS,
        render_board,
    )
except ImportError:  # Direct imports when running from inside ``game``
    from board import (
        EMPTY,
        PLAYER_1,
        PLAYER_2,
        PLAYERS,
        get_neighbors,
        opponent_of,
    )
    from capture import (
        CaptureInfo,
        _could_have_closed_loop,
        detect_capture,
        detect_capture_info,
        find_candidate_regions,
        find_enclosed_regions,
        flood_fill_region,
    )
    from engine import DotsGame
    from groups import UnionFind, find_group, merge_groups, rebuild_groups
    from rendering import BLOCKED_SYMBOL, COLORS, SYMBOLS, render_board


__all__ = [
    "BLOCKED_SYMBOL",
    "COLORS",
    "CaptureInfo",
    "DotsGame",
    "EMPTY",
    "PLAYER_1",
    "PLAYER_2",
    "PLAYERS",
    "SYMBOLS",
    "UnionFind",
    "detect_capture",
    "detect_capture_info",
    "find_candidate_regions",
    "find_enclosed_regions",
    "find_group",
    "flood_fill_region",
    "get_neighbors",
    "merge_groups",
    "opponent_of",
    "rebuild_groups",
    "render_board",
]
