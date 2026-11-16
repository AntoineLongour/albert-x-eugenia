"""Unit tests for Lot 3 possession logic."""

import os
import sys

# Ensure project root is in path even when running `python tests/test_possession.py` directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.possession import two_second_rule


def test_dead_ball_priority():
    """Rule 1: Dead-ball frames ('D') must never be overridden or smoothed."""
    raw = ["H", "H", "H", "H"]
    dead = [False, True, True, False]
    labels = two_second_rule(raw, dead, fps=25, threshold_seconds=2.0)
    assert labels[0] == "H"
    assert labels[1] == "D"
    assert labels[2] == "D"
    # After dead ball without restart info, frame 3 has 'H' in raw so 'H' takes it immediately
    assert labels[3] == "H"


def test_short_opponent_touch_ignored():
    """Rule 2: An isolated opponent touch (< 2s / 50 frames) must not cause a turnover."""
    fps = 25
    raw = ["H"] * 30 + ["A"] * 15 + ["H"] * 30
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=fps, threshold_seconds=2.0)
    assert all(label == "H" for label in labels)


def test_turnover_with_intermittent_dribble_and_nan():
    """Rule 3: Elapsed time counts across NaN/empty frames during an opponent dribble."""
    fps = 25
    # 10 frames of H, then A touches at frame 10, dribbles (NaN for 48 frames), touches at frame 59
    # Elapsed span: 59 - 10 + 1 = 50 frames (2.0s) -> turnover confirmed and backdated to frame 10
    raw = np.array(["H"] * 10 + ["A"] + [np.nan] * 48 + ["A"], dtype=object)
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=fps, threshold_seconds=2.0)

    assert list(labels[:10]) == ["H"] * 10
    assert list(labels[10:]) == ["A"] * 50


def test_missing_ball_maintains_possession():
    """Rule 4: Occluded or loose ball (NaN / '') must forward-fill existing possession."""
    raw = np.array(["H"] * 10 + [np.nan] * 40 + ["H"] * 10, dtype=object)
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=25, threshold_seconds=2.0)
    assert all(label == "H" for label in labels)


def test_restart_with_known_team():
    """Rule 5a: After a dead ball, restart frame takes the awarded team directly."""
    raw = ["H"] * 10 + [""] * 10 + [""] * 20
    dead = [False] * 10 + [True] * 10 + [False] * 20
    restarts = {20: "A"}  # First live frame after dead ball (frames 10..19)
    labels = two_second_rule(raw, dead, restarts=restarts, fps=25, threshold_seconds=2.0)

    assert list(labels[:10]) == ["H"] * 10
    assert list(labels[10:20]) == ["D"] * 10
    assert list(labels[20:]) == ["A"] * 20


def test_restart_without_info_resets_to_first_touch():
    """Rule 5b: After a dead ball with no restart info (e.g. throw-in), do NOT keep old team."""
    raw = ["H"] * 10 + [""] * 5 + ["", "", "A", "A"]
    dead = [False] * 10 + [True] * 5 + [False, False, False, False]
    labels = two_second_rule(raw, dead, restarts=None, fps=25, threshold_seconds=2.0)

    assert list(labels[:10]) == ["H"] * 10
    assert list(labels[10:15]) == ["D"] * 5
    # Frames 15-16 have no touch yet -> must not inherit 'H'
    assert labels[15] == "D"
    assert labels[16] == "D"
    # Frame 17: 'A' touches first -> 'A' gets possession immediately without 2s delay
    assert labels[17] == "A"
    assert labels[18] == "A"


def test_half_time_resets_possession():
    """Rule 6: Possession from the 1st half must never carry over into the 2nd half."""
    raw = ["H"] * 10 + [""] * 5 + ["", "", "A", "A"]
    dead = [False] * 10 + [True] * 5 + [False, False, False, False]
    labels = two_second_rule(
        raw, dead, fps=25, threshold_seconds=2.0, half_time_range=(10, 15)
    )

    assert list(labels[:10]) == ["H"] * 10
    assert list(labels[10:15]) == ["D"] * 5
    assert labels[15] == "D"
    assert labels[16] == "D"
    assert labels[17] == "A"
    assert labels[18] == "A"


if __name__ == "__main__":
    test_dead_ball_priority()
    test_short_opponent_touch_ignored()
    test_turnover_with_intermittent_dribble_and_nan()
    test_missing_ball_maintains_possession()
    test_restart_with_known_team()
    test_restart_without_info_resets_to_first_touch()
    test_half_time_resets_possession()
    print("All Lot 3 tests passed successfully!")