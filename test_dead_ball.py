import numpy as np
from src.dead_ball import (out_of_pitch, ball_speed, drop_short_runs,
                           find_restart, set_piece_dead,
                           find_kickoff, before_kickoff, dead_ball,
                           runs, still_ball_stoppages)

L, W = 105.3, 68.0      # pitch size in metres
NAN = np.nan
F, T = False, True


# ---------- out_of_pitch ----------

def test_ball_in_the_middle_is_in_play():
    x = np.array([52.0]); y = np.array([34.0])
    assert not out_of_pitch(x, y, L, W).any()


def test_ball_on_the_touchline_is_in_play():
    # centre 5 cm beyond the line: less than the radius, still partly on the pitch
    x = np.array([40.0]); y = np.array([W + 0.05])
    assert not out_of_pitch(x, y, L, W).any()


def test_ball_out_for_a_throw_in():
    x = np.array([40.0]); y = np.array([W + 0.50])
    assert out_of_pitch(x, y, L, W).all()


def test_ball_out_behind_the_left_goal():
    x = np.array([-0.30]); y = np.array([34.0])
    assert out_of_pitch(x, y, L, W).all()


def test_ball_out_behind_the_right_goal():
    x = np.array([L + 0.30]); y = np.array([34.0])
    assert out_of_pitch(x, y, L, W).all()


def test_missing_ball_is_not_out():
    x = np.array([NAN]); y = np.array([NAN])
    assert not out_of_pitch(x, y, L, W).any()


def test_several_frames_at_once():
    x = np.array([52.0, 40.0, NAN, L + 1.0])
    y = np.array([34.0, W + 1.0, NAN, 34.0])
    result = out_of_pitch(x, y, L, W)
    assert list(result) == [False, True, False, True]


def test_ball_on_the_goal_line_is_in_play():
    # centre 5 cm behind the goal line: less than the radius, still partly on the pitch
    x = np.array([L + 0.05, -0.05])   # right goal, then left goal
    y = np.array([34.0, 34.0])        # in front of the goal
    assert not out_of_pitch(x, y, L, W).any()


# ---------- ball_speed ----------

def test_speed_first_frame_unknown():
    # no frame before the first one: we cannot know
    x = np.array([30.0, 30.0]); y = np.array([10.0, 10.0])
    assert np.isnan(ball_speed(x, y)[0])


def test_speed_placed_ball_is_zero():
    x = np.array([30.0, 30.0, 30.0]); y = np.array([10.0, 10.0, 10.0])
    assert np.allclose(ball_speed(x, y)[1:], [0.0, 0.0])


def test_speed_straight_pass():
    # 0.5 m per frame, 25 frames per second -> 12.5 m/s
    x = np.array([30.0, 30.5, 31.0]); y = np.array([10.0, 10.0, 10.0])
    assert np.allclose(ball_speed(x, y)[1:], [12.5, 12.5])


def test_speed_diagonal_pass():
    # 0.4 m in x and 0.3 m in y: Pythagoras -> 0.5 m per frame -> 12.5 m/s
    x = np.array([30.0, 30.4]); y = np.array([10.0, 10.3])
    assert np.isclose(ball_speed(x, y)[1], 12.5)


def test_speed_ball_disappears():
    # ball unseen on frame 2: speed unknown on 2 (no position) and on 3 (no previous position)
    x = np.array([30.0, 30.5, NAN, 31.5]); y = np.array([10.0, 10.0, NAN, 10.0])
    v = ball_speed(x, y)
    assert np.isclose(v[1], 12.5)
    assert np.isnan(v[2]) and np.isnan(v[3])


def test_speed_same_length_as_positions():
    # one speed per frame, otherwise frames would be shifted
    x = np.zeros(10); y = np.zeros(10)
    assert len(ball_speed(x, y)) == 10


# ---------- drop_short_runs ----------
# Small minimum length (3 frames) so the arrays stay readable.

def no_protection(n):
    return np.zeros(n, dtype=bool)


def test_short_run_dropped():
    dead = np.array([F, T, T, F, F])               # run of 2 < 3
    result = drop_short_runs(dead, 3, no_protection(5))
    assert list(result) == [F, F, F, F, F]


def test_long_run_kept():
    dead = np.array([F, T, T, T, T, F])            # run of 4 >= 3
    result = drop_short_runs(dead, 3, no_protection(6))
    assert list(result) == [F, T, T, T, T, F]


def test_run_exactly_min_length_kept():
    dead = np.array([F, T, T, T, F])               # run of 3 = 3: kept
    result = drop_short_runs(dead, 3, no_protection(5))
    assert list(result) == [F, T, T, T, F]


def test_short_protected_run_kept():
    dead =    np.array([F, T, T, F])
    protect = np.array([F, T, F, F])               # one protected frame inside the run
    result = drop_short_runs(dead, 3, protect)
    assert list(result) == [F, T, T, F]


def test_several_mixed_runs():
    dead = np.array([T, F, T, T, T, T, F, T, T, F])
    #                └1┘   └───4────┘    └2─┘
    result = drop_short_runs(dead, 3, no_protection(10))
    assert list(result) == [F, F, T, T, T, T, F, F, F, F]


def test_runs_touching_the_edges():
    # a short run at the very start, a long one up to the very end
    dead = np.array([T, F, F, T, T, T])
    result = drop_short_runs(dead, 3, no_protection(6))
    assert list(result) == [F, F, F, T, T, T]


def test_all_in_play_stays_in_play():
    dead = np.zeros(8, dtype=bool)
    assert not drop_short_runs(dead, 3, no_protection(8)).any()


def test_input_array_not_modified():
    dead = np.array([F, T, F])
    drop_short_runs(dead, 3, no_protection(3))
    assert list(dead) == [F, T, F]                 # the original is untouched


# ---------- find_restart ----------
# Here we pass SPEEDS (m/s) directly, not positions: it is easier to read.
# Parameters: still if v <= 2 m/s, placed at least 5 frames before the kick.

def test_restart_ball_placed_then_kicked():
    #             0    1  2  3  4  5   6   7
    v = np.array([NAN, 0, 0, 0, 0, 0, 12, 12])
    assert find_restart(v, 0, 8) == 6


def test_restart_ball_carried_then_placed_then_thrown():
    # frames 0-2: the player runs with the ball; 3-8: holds it; 9: throws it
    v = np.array([4, 4, 3, 0, 0, 0, 0, 0, 0, 12, 12], dtype=float)
    assert find_restart(v, 0, 11) == 9


def test_restart_never_placed_returns_end():
    # the ball keeps moving: no visible restart
    v = np.array([5, 6, 7, 8, 9], dtype=float)
    assert find_restart(v, 0, 5) == 5


def test_restart_ball_unseen_returns_end():
    v = np.full(6, NAN)
    assert find_restart(v, 0, 6) == 6


def test_restart_placed_too_briefly_does_not_count():
    # placed 3 frames (< 5) then touched: not a restart; placed again 5 frames then kicked: restart
    v = np.array([0, 0, 0, 12, 0, 0, 0, 0, 0, 12], dtype=float)
    assert find_restart(v, 0, 10) == 9


def test_restart_two_short_still_spells_do_not_add_up():
    # 3 frames placed + movement + 2 frames placed: never 5 in a row
    v = np.array([0, 0, 0, 12, 0, 0, 12], dtype=float)
    assert find_restart(v, 0, 7) == 7


def test_restart_unseen_ball_breaks_stillness():
    # 3 frames placed, 1 frame unseen, 2 frames placed, kick: never 5 seen frames in a row
    v = np.array([0, 0, 0, NAN, 0, 0, 12], dtype=float)
    assert find_restart(v, 0, 7) == 7


def test_restart_speed_exactly_at_threshold_counts_as_placed():
    # exactly 2 m/s = still placed (tracking jitters a little)
    v = np.array([2, 2, 2, 2, 2, 12], dtype=float)
    assert find_restart(v, 0, 6) == 5


def test_restart_only_searched_inside_the_window():
    # a restart BEFORE the window (frame 6) and one AFTER it (frame 16) are ignored
    v = np.array([0, 0, 0, 0, 0, 0, 12,         # before the window
                  0, 0, 0, 0, 0, 0, 12, 12,     # window: frames 7 to 14, restart on 13
                  0, 12], dtype=float)          # after the window
    assert find_restart(v, 7, 15) == 13


# ---------- set_piece_dead ----------
# This function takes POSITIONS: two helpers build made-up situations.

def placed_then_kicked(n_placed, n_kicked, start=40.0):
    """Ball placed for n_placed frames at x=start, then kicked (0.5 m per frame = 12.5 m/s)."""
    x = np.concatenate([np.full(n_placed, start),
                        start + 0.5 * np.arange(1, n_kicked + 1)])
    y = np.full(len(x), 30.0)
    return x, y


def tag(start, end, code="SET PIECES", type_="Throw-in"):
    """A Smart Tagging entry, in the format of load_smart_tagging.load()."""
    return {"code": code, "start_frame": start, "end_frame": end,
            "tags": {"Type": type_, "Team": "N/A"}}


def test_throw_in_dead_until_restart():
    x, y = placed_then_kicked(7, 3)                # frames 0-6 placed, kicked on 7
    dead = set_piece_dead(x, y, [tag(0, 10)], n=10)
    assert list(dead) == [T] * 7 + [F] * 3


def test_other_phase_ignored():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(0, 10, code="BUILD UP")], n=10)
    assert not dead.any()


def test_case_and_spaces_ignored():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(0, 10, code="  Set Pieces ")], n=10)
    assert list(dead) == [T] * 7 + [F] * 3


def test_missing_code_does_not_crash():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(0, 10, code=None)], n=10)
    assert not dead.any()


def test_before_the_window_stays_in_play():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(2, 10)], n=10)
    assert list(dead) == [F, F] + [T] * 5 + [F] * 3


def test_two_set_pieces():
    x1, y1 = placed_then_kicked(7, 3, start=40.0)   # frames 0-9, restart on 7
    x2, y2 = placed_then_kicked(7, 3, start=60.0)   # frames 10-19, restart on 17
    x, y = np.concatenate([x1, x2]), np.concatenate([y1, y2])
    dead = set_piece_dead(x, y, [tag(0, 10), tag(10, 20)], n=20)
    assert list(dead) == [T] * 7 + [F] * 3 + [T] * 7 + [F] * 3


def test_window_overflowing_the_match():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(-5, 50)], n=10)
    assert list(dead) == [T] * 7 + [F] * 3


def test_window_entirely_outside_the_match():
    x, y = placed_then_kicked(7, 3)
    dead = set_piece_dead(x, y, [tag(15, 20)], n=10)
    assert not dead.any()


def test_ball_unseen_for_the_whole_window():
    x = np.full(8, NAN); y = np.full(8, NAN)
    dead = set_piece_dead(x, y, [tag(0, 8)], n=8)
    assert dead.all()


# ---------- find_kickoff ----------

def kickoff_tag(start, end, type_="Kick Off (Start)", half="2nd Half"):
    """A Smart Tagging kickoff entry, in the right format."""
    return {"code": "SET PIECES", "start_frame": start, "end_frame": end,
            "tags": {"Type": type_, "Half": half, "Team": "Barcelona"}}


def test_kickoff_finds_the_kick():
    x, y = placed_then_kicked(7, 3)                # placed on the centre spot, kicked on 7
    tags = [kickoff_tag(0, 10)]
    assert find_kickoff(tags, ball_speed(x, y), 10, "2nd Half") == 7


def test_kickoff_ignores_the_other_half():
    x, y = placed_then_kicked(7, 3)
    tags = [kickoff_tag(0, 10, half="1st Half")]
    assert find_kickoff(tags, ball_speed(x, y), 10, "2nd Half") is None


def test_kickoff_ignores_other_set_pieces():
    x, y = placed_then_kicked(7, 3)
    tags = [kickoff_tag(0, 10, type_="Throw-in")]
    assert find_kickoff(tags, ball_speed(x, y), 10, "2nd Half") is None


def test_kickoff_ignores_kickoff_after_a_goal():
    x, y = placed_then_kicked(7, 3)
    tags = [kickoff_tag(0, 10, type_="Kick Off (After Goal)")]
    assert find_kickoff(tags, ball_speed(x, y), 10, "2nd Half") is None


def test_kickoff_case_and_spaces_ignored():
    x, y = placed_then_kicked(7, 3)
    tags = [kickoff_tag(0, 10, type_="KICK OFF (START)", half=" 2nd half ")]
    assert find_kickoff(tags, ball_speed(x, y), 10, "2nd Half") == 7


def test_kickoff_without_any_tag():
    x, y = placed_then_kicked(7, 3)
    assert find_kickoff([], ball_speed(x, y), 10, "2nd Half") is None


# ---------- before_kickoff ----------

def test_wait_before_kickoff_is_dead():
    dead = before_kickoff(10, start=3, kickoff=7)
    assert list(dead) == [F, F, F, T, T, T, T, F, F, F]


def test_wait_without_kickoff_found():
    dead = before_kickoff(10, start=3, kickoff=None)
    assert not dead.any()


def test_inconsistent_wait_ignored():
    # kickoff BEFORE the start of the half: impossible, mark nothing
    dead = before_kickoff(10, start=7, kickoff=3)
    assert not dead.any()


def test_wait_right_length():
    assert len(before_kickoff(10, start=3, kickoff=7)) == 10


# ---------- dead_ball (putting it all together) ----------

def make_trk(x, y):
    """The dictionary lot 4 will provide, with only what we need."""
    return {"ball_x": np.array(x, dtype=float), "ball_y": np.array(y, dtype=float)}


def test_open_play_all_in_play():
    x = 40 + 0.5 * np.arange(10); y = np.full(10, 30.0)      # a pass in midfield
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3)
    assert not dead.any()


def test_unseen_ball_is_not_dead():
    # unseen != dead: lot 3 keeps the current team
    x = np.full(10, NAN); y = np.full(10, NAN)
    assert not dead_ball(make_trk(x, y), [], min_stoppage=3).any()


def test_long_ball_out_kept():
    x = np.full(10, 40.0); y = np.full(10, 30.0)
    y[3:8] = W + 1.0                                        # out for 5 frames
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3)
    assert list(dead) == [F] * 3 + [T] * 5 + [F] * 2


def test_too_short_ball_out_dropped():
    x = np.full(10, 40.0); y = np.full(10, 30.0)
    y[3:5] = W + 1.0                                        # out for 2 frames < 3: noise
    assert not dead_ball(make_trk(x, y), [], min_stoppage=3).any()


def test_ball_out_then_throw_in_form_a_single_stoppage():
    #  0-1: in play | 2-3: out | 4-9: placed on the line | 10-11: thrown
    x = [40, 40.5, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41]
    y = [30, 30, W + 1, W + 1, W, W, W, W, W, W, W - 0.5, W - 1]
    tags = [tag(3, 12)]                                     # throw-in window
    dead = dead_ball(make_trk(x, y), tags, min_stoppage=3)
    # alone, the ball out (2 frames) would be dropped; joined to the throw-in, it is kept
    assert list(dead) == [F, F] + [T] * 8 + [F, F]


def test_kickoff_wait_protected_even_if_short():
    x, y = placed_then_kicked(7, 3)                         # placed 0-6, kicked on 7
    tags = [kickoff_tag(1, 10)]                             # kickoff found on 7
    dead = dead_ball(make_trk(x, y), tags, second_half_start=5, min_stoppage=25)
    # stoppage of 6 frames < 25, but it contains the kickoff wait: kept
    assert list(dead) == [F] + [T] * 6 + [F] * 3


def test_kickoff_not_found_does_not_crash():
    x = 40 + 0.5 * np.arange(10); y = np.full(10, 30.0)
    dead = dead_ball(make_trk(x, y), [], second_half_start=2, min_stoppage=3)
    assert not dead.any()


def test_contract_simple_two_argument_call():
    # this is how main.py will call it: dead_ball(trk, tags)
    x = 40 + 0.5 * np.arange(10); y = np.full(10, 30.0)
    dead = dead_ball(make_trk(x, y), [])
    assert len(dead) == 10 and dead.dtype == bool


# ---------- runs ----------

def test_runs_finds_start_and_end():
    assert runs(np.array([F, T, T, F, T])) == [(1, 3), (4, 5)]


def test_runs_none():
    assert runs(np.array([F, F, F])) == []


def test_runs_all_true():
    assert runs(np.array([T, T, T])) == [(0, 3)]


# ---------- still_ball_stoppages (rule E) ----------
# Small duration (3 frames) for readability; in the real code, 3 s = 75 frames.

def test_foul_ball_placed_for_long_then_kicked():
    v = np.array([12, 0, 0, 0, 0, 12, 12], dtype=float)   # placed on 1-4, kicked on 5
    assert list(still_ball_stoppages(v, 3)) == [F, T, T, T, T, F, F]


def test_ball_still_briefly_stays_in_play():
    v = np.array([12, 0, 0, 12, 12], dtype=float)          # 2 frames < 3: a ball control
    assert not still_ball_stoppages(v, 3).any()


def test_exactly_the_duration_counts_as_stoppage():
    v = np.array([12, 0, 0, 0, 12], dtype=float)           # 3 frames = 3
    assert list(still_ball_stoppages(v, 3)) == [F, T, T, T, F]


def test_unseen_ball_breaks_stillness():
    v = np.array([0, 0, NAN, 0, 0], dtype=float)           # never 3 SEEN frames in a row
    assert not still_ball_stoppages(v, 3).any()


def test_ball_always_moving():
    v = np.full(8, 10.0)
    assert not still_ball_stoppages(v, 3).any()


# ---------- dead_ball with rule E ----------

def test_dead_ball_detects_an_untagged_foul():
    x, y = placed_then_kicked(5, 3)                        # placed 0-4, kicked on 5, no tag
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3, min_still=4)
    assert list(dead) == [F, T, T, T, T, F, F, F]

def test_ball_lost_in_the_stands_stays_dead_until_the_throw():
    #  0-1: in play | 2-3: out | 4-7: unseen (stands) | 8-10: new ball held behind the line | 11-12: thrown in
    x = [40, 40, 40, 40, NAN, NAN, NAN, NAN, 40, 40, 40, 40, 40]
    y = [30, 30, W + 1, W + 2, NAN, NAN, NAN, NAN, W + 0.3, W + 0.3, W + 0.3, W - 0.3, W - 0.9]
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3)
    assert list(dead) == [F, F] + [T] * 9 + [F, F]
  
def test_new_ball_placed_inside_stays_dead_until_kicked():
    # corner: out behind the goal line, unseen, new ball placed in the arc (inside), then kicked
    x = [L - 5, L - 4, L + 1, NAN, NAN, L - 0.3, L - 0.3, L - 0.3, L - 0.9, L - 1.5]
    y = [30, 30, 30, NAN, NAN, 0.3, 0.3, 0.3, 0.7, 1.1]
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3)
    assert list(dead) == [F, F] + [T] * 6 + [F, F]
  
def test_ghost_out_for_one_frame_is_dropped():
    # one frame wrongly seen out, then the ball is seen moving on the pitch again
    x = 40 + 0.5 * np.arange(10); y = np.full(10, 30.0)
    y[4] = W + 1.0
    dead = dead_ball(make_trk(x, y), [], min_stoppage=3)
    assert not dead.any()

