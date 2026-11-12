"""Unit tests for Lot 3 possession logic."""

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
    assert labels[3] == "H"


def test_short_opponent_touch_ignored():
    """Rule 2: An isolated opponent touch (< 2s / 50 frames) must not cause a turnover."""
    fps = 25
    # H in possession, A touches for 15 frames (0.6s), then H touches again
    raw = ["H"] * 30 + ["A"] * 15 + ["H"] * 30
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=fps, threshold_seconds=2.0)
    assert all(label == "H" for label in labels)


def test_turnover_with_retroactive_boundary_shift():
    """Rule 3: Validated turnover must backdate the boundary to the initial contact."""
    fps = 25
    # 20 frames of H, followed by 60 frames of A (> 50 frames / 2.0s)
    raw = ["H"] * 20 + ["A"] * 60
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=fps, threshold_seconds=2.0)

    # Frame 0 to 19 must be H
    assert list(labels[:20]) == ["H"] * 20
    # From frame 20 onwards, boundary is backdated to A
    assert list(labels[20:]) == ["A"] * 60


def test_missing_ball_maintains_possession():
    """Rule 4: Occluded or loose ball ('') must forward-fill existing possession."""
    raw = ["H"] * 10 + [""] * 40 + ["H"] * 10
    dead = np.zeros(len(raw), dtype=bool)
    labels = two_second_rule(raw, dead, fps=25, threshold_seconds=2.0)
    assert all(label == "H" for label in labels)


def test_restart_team_allocation():
    """Rule 5: After a dead ball, restart frame takes the awarded team directly."""
    raw = ["H"] * 10 + [""] * 10 + [""] * 20
    dead = [False] * 10 + [True] * 10 + [False] * 20
    restarts = {20: "A"}  # Set piece restarted by Away at frame 20
    labels = two_second_rule(raw, dead, restarts=restarts, fps=25, threshold_seconds=2.0)

    assert list(labels[:10]) == ["H"] * 10
    assert list(labels[10:20]) == ["D"] * 10
    assert list(labels[20:]) == ["A"] * 20


def test_half_time_resets_possession():
    """Rule 6: Possession from the 1st half must never carry over into the 2nd half."""
    # Simulate frames around a custom half-time window (frames 10 to 15)
    raw = ["H"] * 10 + [""] * 5 + ["", "", "A", "A"]
    dead = [False] * 10 + [True] * 5 + [False, False, False, False]
    labels = two_second_rule(
        raw, dead, fps=25, threshold_seconds=2.0, half_time_range=(10, 15)
    )

    # First half ends with H
    assert list(labels[:10]) == ["H"] * 10
    # Half-time break is D
    assert list(labels[10:15]) == ["D"] * 5
    # Start of 2nd half before first contact must NOT inherit 'H'
    assert labels[15] == "D"
    assert labels[16] == "D"
    # Once 'A' touches the ball, 'A' takes possession immediately
    assert labels[17] == "A"
    assert labels[18] == "A"


if __name__ == "__main__":
    test_dead_ball_priority()
    test_short_opponent_touch_ignored()
    test_turnover_with_retroactive_boundary_shift()
    test_missing_ball_maintains_possession()
    test_restart_team_allocation()
    test_half_time_resets_possession()
    print("All Lot 3 tests passed successfully!")