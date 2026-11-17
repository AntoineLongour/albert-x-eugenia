import numpy as np
from games import GAMES, GAME, half_video_blocks

FPS = GAMES[GAME]["fps"]   # 25 frames per second, read from games.py

# Parameters (each one is justified in DECISIONS.md)
BALL_RADIUS = 0.11            # m. Law 9: the ball is out when it has WHOLLY crossed the line
RESTART_SPEED = 2.0           # m/s. Brisk walking pace: below it, nobody is really playing the ball
MIN_STILL_BEFORE_RESTART = 5  # frames = 0.2 s. Before a restart, the ball is placed or held still
MIN_STILL_STOPPAGE = 75       # frames = 3 s. In open play the ball is never still for 3 s: the referee has stopped play
MIN_STOPPAGE = 25             # frames = 1 s. A real stoppage lasts longer; anything shorter is tracking noise

SET_PIECE_CODE = "set pieces"     # Smart Tagging code for set pieces (lower case)
KICKOFF_TYPE = "kick off (start)" # kickoff that starts a half (not after a goal)


def out_of_pitch(x, y, length, width):
    """True where the ball is beyond a line by at least its radius. Missing ball (NaN) -> False."""
    with np.errstate(invalid="ignore"):
        return ((x < -BALL_RADIUS) | (x > length + BALL_RADIUS) |
                (y < -BALL_RADIUS) | (y > width + BALL_RADIUS))


def ball_speed(x, y):
    """Ball speed in m/s, between each frame and the previous one.
    NaN on frame 0 and wherever either position is missing."""
    v = np.full(len(x), np.nan)
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    return v


def runs(mask):
    """List of (start, end) for every run of True, end excluded."""
    m = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))

def hold_until_back_in_play(out, x, y, v):
    """Rule A+: once the ball has been seen out, play stays stopped until the ball is seen
    back INSIDE the pitch AND moving faster than RESTART_SPEED (a kick or a throw).
    Covers a ball lost in the stands, a new ball from a ball boy, a ball carried to the spot."""
    seen = ~np.isnan(x) & ~np.isnan(y)
    with np.errstate(invalid="ignore"):
        back_in_play = seen & ~out & (v > RESTART_SPEED)   # NaN speed -> False
    held = out.copy()
    stopped = False
    for i in range(len(out)):
        if out[i]:
            stopped = True
        elif stopped and back_in_play[i]:
            stopped = False
        held[i] = stopped
    return held

def still_ball_stoppages(v, duration):
    """Rule E: True on every run of at least `duration` frames where the ball is SEEN still.
    The stoppage ends on the frame where the ball moves again (excluded)."""
    with np.errstate(invalid="ignore"):
        still = v <= RESTART_SPEED              # NaN -> False: an unseen ball is not a placed ball
    dead = np.zeros(len(v), dtype=bool)
    for start, end in runs(still):
        if end - start >= duration:
            dead[start:end] = True
    return dead


def drop_short_runs(dead, min_length, protect):
    """Set back to False every run of True shorter than min_length,
    unless at least one of its frames is protected. Does not modify `dead`."""
    out = dead.copy()
    i, n = 0, len(out)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_length and not protect[i:j].any():
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def find_restart(v, start, end):
    """First frame in [start, end) where the ball moves again after being still
    for at least MIN_STILL_BEFORE_RESTART frames in a row. Returns `end` if that never happens."""
    still_for = 0
    for i in range(start, end):
        if np.isnan(v[i]):
            still_for = 0                    # ball not seen: we cannot say it is placed
        elif v[i] <= RESTART_SPEED:
            still_for += 1                   # placed: keep counting
        else:                                # it moves
            if still_for >= MIN_STILL_BEFORE_RESTART:
                return i                     # it moves AFTER being placed: this is the restart
            still_for = 0                    # it was already moving (carried, controlled): start over
    return end


def set_piece_dead(x, y, tags, n):
    """True from the start of every SET PIECES window up to the restart (excluded)."""
    dead = np.zeros(n, dtype=bool)
    v = ball_speed(x, y)
    for t in tags:
        if (t["code"] or "").strip().lower() != SET_PIECE_CODE:
            continue
        start = max(0, t["start_frame"])
        end = min(n, t["end_frame"])
        if start >= end:
            continue
        restart = find_restart(v, start, end)
        dead[start:restart] = True
    return dead


def find_kickoff(tags, v, n, half):
    """Kickoff frame of a half, found in the data:
    the Smart Tagging "Kick Off (Start)" window, then the placed ball moving again.
    None if no tag matches."""
    for t in tags:
        code = (t["code"] or "").strip().lower()
        type_ = (t["tags"].get("Type") or "").strip().lower()
        tag_half = (t["tags"].get("Half") or "").strip().lower()
        if (code == SET_PIECE_CODE and type_ == KICKOFF_TYPE
                and tag_half == half.strip().lower()):
            start = max(0, t["start_frame"])
            end = min(n, t["end_frame"])
            return find_restart(v, start, end)
    return None


def before_kickoff(n, start, kickoff):
    """True from `start` (included) to `kickoff` (excluded): waiting for the kickoff.
    Nothing if the kickoff is unknown or inconsistent."""
    dead = np.zeros(n, dtype=bool)
    if kickoff is not None and start < kickoff:
        dead[start:kickoff] = True
    return dead


def half_start(half):
    """First graded video frame of a half (1 or 2), read from games.py."""
    for block_half, first, last in half_video_blocks(GAME):
        if block_half == half:
            return first
    raise ValueError("half {} not found in games.py".format(half))


def dead_ball(trk, tags, second_half_start=None, min_stoppage=MIN_STOPPAGE,
              min_still=MIN_STILL_STOPPAGE):
    """For every video frame: True if the ball is dead, False otherwise.

    The contract with the other lots: dead_ball(trk, tags).
    The other parameters are only for tests; by default games.py, 1 s and 3 s."""
    x, y = trk["ball_x"], trk["ball_y"]
    n = len(x)
    length, width = GAMES[GAME]["pitch"]
    if second_half_start is None:
        second_half_start = half_start(2)
    v = ball_speed(x, y)

    waiting = before_kickoff(n, second_half_start, find_kickoff(tags, v, n, "2nd Half"))

    out = hold_until_back_in_play(out_of_pitch(x, y, length, width), x, y, v)
    dead = (out                                      
            | set_piece_dead(x, y, tags, n)          
            | waiting                                
            | still_ball_stoppages(v, min_still))    
                  
    return drop_short_runs(dead, min_stoppage, protect=waiting)
