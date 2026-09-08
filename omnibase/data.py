"""Getting recorded gripper poses in, without caring much where they came from.

The library only ever needs two arrays per hand: ``pos`` (N, 3) in metres and ``quat`` (N, 4) as
``(x, y, z, w)``, both in one fixed world frame. If you already have those, skip this module
entirely -- everything else takes them directly.

What is here is a convenience for the common case: a LeRobot-style parquet whose action column
packs each hand as a contiguous ``[x, y, z, qx, qy, qz, qw, grip]`` block. That is the layout
UMI-style hand-held rigs tend to produce. If yours differs, read it however you like and pass
the arrays.

pandas is imported lazily so the rest of the library stays a numpy-and-scipy dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def from_parquet(path, episode=None, hands=(0, 1), width=8, column="action",
                 episode_column="episode_index"):
    """Per-hand ``(pos, quat, grip)`` out of a LeRobot-style parquet.

    Args:
        path: the parquet file, or a dataset directory containing ``data/**/*.parquet``.
        episode: which episode, or None for the longest one present.
        hands: which blocks of the action vector to pull, in order.
        width: how wide one hand's block is. The first seven are taken as pose, the eighth (if
            present) as the gripper command.

    Returns:
        ``(episode_index, [(pos, quat, grip), ...])``.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("data/**/*.parquet"))
        if not found:
            raise FileNotFoundError(f"no data/**/*.parquet under {path}")
        df = pd.concat([pd.read_parquet(f) for f in found], ignore_index=True)
    else:
        df = pd.read_parquet(path)
    if column not in df.columns:
        raise KeyError(f"{column!r} is not a column of {path}; have {list(df.columns)}")

    act = np.stack(df[column].to_numpy()).astype(np.float64)
    if episode_column in df.columns:
        epi = df[episode_column].to_numpy()
        if episode is None:
            episode = int(np.bincount(epi).argmax())
        sel = np.where(epi == episode)[0]
        if not len(sel):
            raise ValueError(f"episode {episode} is not in {path} (have 0..{int(epi.max())})")
    else:
        episode, sel = 0, np.arange(len(act))

    need = (max(hands) + 1) * width
    if act.shape[1] < need:
        raise ValueError(f"{column} is {act.shape[1]} wide; hands={hands} at width {width} "
                         f"needs {need}. Pass hands=/width= to match your layout.")
    out = []
    for h in hands:
        c = h * width
        grip = act[sel, c + 7] if width > 7 else np.zeros(len(sel))
        out.append((act[sel, c:c + 3], act[sel, c + 3:c + 7], grip))
    return episode, out


def normalise_quats(quat, atol=1e-3):
    """Unit-length ``(x, y, z, w)``, with a loud failure if they were not close to it.

    A quaternion column that is not normalised is nearly always a sign that the layout was read
    wrongly -- a shifted offset, or (w, x, y, z) mistaken for (x, y, z, w). Rescaling silently
    would turn that into a subtly wrong answer instead of an error.
    """
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q, axis=-1)
    if np.abs(norm - 1.0).max() > atol:
        raise ValueError(
            f"quaternions are not unit length (worst |q| = {norm.max():.4f}, "
            f"{np.abs(norm - 1.0).argmax()} of {len(q)}). Check the column offset, and check "
            "the order really is (x, y, z, w) and not (w, x, y, z).")
    return q / norm[..., None]
