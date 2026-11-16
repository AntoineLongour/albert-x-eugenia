"""Lot 3: Temporal stability via the 2-second rule and state machine.

Enforces football possession continuity:
1. Absolute priority for dead ball ('D') frames.
2. Strict reset across half-time boundaries (never carry possession from 1st to 2nd half).
3. Forward-filling possession when the ball is loose, aerial, or occluded.
4. Turnovers require sustained opponent control (>= 2 seconds elapsed from first to latest contact).
5. Retroactive boundary alignment to the opponent's initial contact upon turnover confirmation.
6. Immediate possession assignment on restarts (from `restarts` dict, else the last team seen
   in contact during the stoppage, else first live touch).
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np


def two_second_rule(
    raw_labels: Sequence[Optional[str]],
    dead_ball: Sequence[bool],
    restarts: Optional[Dict[int, str]] = None,
    fps: int = 25,
    threshold_seconds: float = 2.0,
    half_time_range: Optional[Tuple[int, int]] = (90652, 110808),
) -> np.ndarray:
    """Apply the 2-second elapsed control threshold to stabilize possession sequences.

    Parameters
    ----------
    raw_labels : Sequence[Optional[str]]
        Candidate team labels ('H', 'A', or '' / None / np.nan) per video frame.
    dead_ball : Sequence[bool]
        Boolean array indicating whether the ball is dead ('D') for each frame.
    restarts : Optional[Dict[int, str]]
        Mapping of the first live frame after a dead ball to team ('H' or 'A').
    fps : int, default=25
        Video frame rate (anchor FPS).
    threshold_seconds : float, default=2.0
        Minimum elapsed duration between first and confirming opponent contact.
    half_time_range : Optional[Tuple[int, int]], default=(90652, 110808)
        Start and end video frames of the half-time break where state must reset.

    Returns
    -------
    np.ndarray
        Array of length N containing finalized labels: 'H', 'A', or 'D'.
    """
    n_frames = len(raw_labels)
    threshold_frames = int(fps * threshold_seconds)

    labels = np.full(n_frames, "D", dtype="<U1")
    current_team: Optional[str] = None

    # Tracking pending opponent challenge / turnover attempt
    candidate_opponent: Optional[str] = None
    opponent_first_contact: Optional[int] = None

    # Last team seen in contact while the ball was dead (thrower / kicker of the restart)
    last_dead_contact: Optional[str] = None

    for i in range(n_frames):
        # 0. Half-time barrier: never carry possession across halves
        if half_time_range is not None and half_time_range[0] <= i < half_time_range[1]:
            current_team = None
            candidate_opponent = None
            opponent_first_contact = None
            last_dead_contact = None
            labels[i] = "D"
            continue

        # 1. Dead-ball state overrides everything
        if dead_ball[i]:
            labels[i] = "D"
            candidate_opponent = None
            opponent_first_contact = None
            if raw_labels[i] in ("H", "A"):
                last_dead_contact = raw_labels[i]
            continue

        # 2. Restart of play following a dead ball (indexed on first live frame i)
        if i > 0 and dead_ball[i - 1] and not dead_ball[i]:
            # The kick/throw is too fast for Lot 1 to see who touched the ball, so fall back
            # on the last team seen in contact during the stoppage.
            if restarts and i in restarts:
                current_team = restarts[i]
            else:
                current_team = last_dead_contact
            last_dead_contact = None
            candidate_opponent = None
            opponent_first_contact = None

        contact = raw_labels[i]

        # Initialisation on the first live contact of a half or unknown restart
        if current_team is None:
            if contact in ("H", "A"):
                current_team = contact
                labels[i] = contact
            else:
                labels[i] = "D"
            continue

        opponent = "A" if current_team == "H" else "H"

        # 3. Opponent is detected in proximity / contact
        if contact == opponent:
            if candidate_opponent != opponent:
                candidate_opponent = opponent
                opponent_first_contact = i

            assert opponent_first_contact is not None
            # Elapsed time >= 2s since first contact: turnover confirmed, backdate boundary
            if i - opponent_first_contact + 1 >= threshold_frames:
                labels[opponent_first_contact : i + 1] = opponent
                current_team = opponent
                candidate_opponent = None
                opponent_first_contact = None
            else:
                # Under evaluation: maintain current team until threshold is met
                labels[i] = current_team

        # 4. Current team re-touches the ball: challenge dismissed
        elif contact == current_team:
            candidate_opponent = None
            opponent_first_contact = None
            labels[i] = current_team

        # 5. Loose ball / occluded / NaN: persistence (timer keeps running if opponent pending)
        else:
            labels[i] = current_team

    return labels


# Contract alias for team compatibility
regle_2s = two_second_rule