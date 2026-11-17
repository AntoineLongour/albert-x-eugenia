"""Albert x Eugenia -- regenerates our submission file from the match data.

    python main.py          ->  writes "Albert x Eugenia.csv" at the root of the repository

Pipeline (the tracking file is read once):
  1. src/contact.py     who touches the ball on each frame: H, A or nobody (lot 1)
  2. src/dead_ball.py   is play stopped on each frame: True / False (lot 2)
  3. src/possession.py  2-second rule -> stable possession H / A / D (lot 3)
  4. frames outside the two graded halves -> X (never scored)

The data is read from data/ (or from the folder in $MTAH_DATA_DIR) and is never stored here.
Same data + same code -> byte-identical file: the SHA-256 printed at the end lets you check it.
"""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402
from src.contact import equipe_brute, lire_tracking  # noqa: E402
from src.dead_ball import dead_ball  # noqa: E402
from src.possession import regle_2s  # noqa: E402

OUTPUT = os.path.join(ROOT, "Albert x Eugenia.csv")


def build_labels(data_dir=None, game=GAME):
    """One label per video frame of the tactical video: H, A, D or X."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_of_track = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)

    ball, players = lire_tracking(raw_path, n_frames, len(team_of_track))   # metres, NaN = unseen

    contact = equipe_brute(ball, players, team_of_track)                      # lot 1
    trk = {"ball_x": ball[:, 0].astype(float), "ball_y": ball[:, 1].astype(float)}
    dead = dead_ball(trk, load_smart_tagging(game, data_dir))                 # lot 2
    labels = np.asarray(regle_2s(contact.to_numpy(), dead), dtype=object)     # lot 3

    graded = np.zeros(n_frames, dtype=bool)
    for _half, first, last in half_video_blocks(game):                        # inclusive bounds
        graded[first:last + 1] = True
    labels[~graded] = "X"
    return labels


def write_csv(labels, path):
    """frame,label -- '\\n' line endings on every OS, so the file is identical everywhere."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        for frame, label in enumerate(labels):
            f.write("{},{}\n".format(frame, label))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    write_csv(build_labels(), OUTPUT)
    print("written:", os.path.relpath(OUTPUT, ROOT))
    print("sha256 :", sha256(OUTPUT))
