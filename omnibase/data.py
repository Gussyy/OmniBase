"""Reading LeRobot datasets, without assuming they look like the one this was written against.

The library itself only ever needs two arrays per hand: ``pos`` (N, 3) in metres and ``quat``
(N, 4) as ``(x, y, z, w)``, in one fixed world frame. If you already have those, ignore this
module and pass them straight in.

What is here is for LeRobot datasets, and it tries to handle the format rather than one dialect
of it. Both on-disk layouts are read:

* **v2.1** -- one parquet per episode, ``data/chunk-000/episode_000000.parquet``
* **v3.0** -- episodes concatenated, ``data/chunk-000/file-000.parquet``, sliced by
  ``episode_index``

and both kinds of action column are understood, because which one you have decides whether this
library has anything to say:

* **end-effector poses** -- columns named ``<hand>_x, _y, _z, _qx, _qy, _qz, _qw`` (plus an
  optional gripper). This is what a hand-held rig records, and it is the case OmniBase is for:
  the poses were flown by a person and no robot was involved, so where a base would have to
  stand is an open question.
* **joint angles** -- columns named after a robot's joints. These came off a robot that already
  had a base, so there is nothing to place. They are still readable: give a ``chain`` and the
  joint angles are run through forward kinematics into poses, which is useful for asking where
  that robot *should* have stood, or for treating one robot's recording as a source for another.

Columns are matched by the names in ``meta/info.json`` rather than by position, so a dataset
with three hands, or a gripper column in an odd place, or no gripper at all, reads correctly.

pandas is imported lazily, so the rest of the library stays numpy-and-scipy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The seven fields that make a pose, in the order this library wants them.
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
#: Names a gripper column might go by, lowercased.
GRIP_FIELDS = ("grip", "gripper", "jaw", "grasp", "width")


@dataclass
class Hand:
    """One gripper's trajectory, however the dataset happened to store it."""

    name: str
    pos: np.ndarray                  # (N, 3) metres
    quat: np.ndarray                 # (N, 4) as (x, y, z, w)
    grip: np.ndarray | None = None   # (N,) whatever the recording called it
    source: str = "pose"             # "pose" if stored as one, "fk" if solved from joints
    joints: np.ndarray | None = None  # (N, n) when source == "fk"


@dataclass
class Episode:
    """What one episode of a LeRobot dataset holds, in this library's terms."""

    index: int
    hands: list = field(default_factory=list)
    fps: float | None = None
    version: str | None = None
    frames: int = 0

    def __iter__(self):
        return iter(self.hands)

    def __len__(self):
        return len(self.hands)


def _info(root: Path):
    p = root / "meta" / "info.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def episodes(path):
    """Every episode index in a dataset, sorted. For sweeping a whole recording session.

    Read from the metadata where there is any, and from the data itself where there is not --
    v2.1 names the episode in each file, v3.0 puts it in a column.
    """
    root = Path(path)
    n = _info(root).get("total_episodes")
    if n:
        return list(range(int(n)))
    files = sorted(root.glob("data/**/*.parquet")) if root.is_dir() else [root]
    named = sorted({int(m.group(1)) for f in files
                    for m in [re.search(r"episode_(\d+)", f.name)] if m})
    if named:
        return named
    import pandas as pd
    seen = set()
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index"]) if files else None
        seen.update(int(v) for v in df["episode_index"].unique())
    return sorted(seen)


def _frame(root: Path, column: str, episode):
    """Every frame of one episode, whichever on-disk layout the dataset uses."""
    import pandas as pd

    files = sorted(root.glob("data/**/*.parquet")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"no data/**/*.parquet under {root}")

    # v2.1 puts one episode per file and names it in the path; reading only that file beats
    # concatenating a hundred of them to throw all but one away.
    if episode is not None:
        named = [f for f in files if f"episode_{int(episode):06d}" in f.name]
        if named:
            files = named
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if len(files) > 1 \
        else pd.read_parquet(files[0])
    if column not in df.columns:
        raise KeyError(f"{column!r} is not a column of {root}; have {list(df.columns)}")

    if "episode_index" in df.columns:
        epi = df["episode_index"].to_numpy()
        if episode is None:
            episode = int(np.bincount(epi).argmax())
        sel = np.where(epi == int(episode))[0]
        if not len(sel):
            raise ValueError(f"episode {episode} is not in {root} "
                             f"(have {int(epi.min())}..{int(epi.max())})")
    else:
        episode, sel = int(episode or 0), np.arange(len(df))
    return int(episode), np.stack(df[column].to_numpy()).astype(np.float64)[sel]


def _split(names):
    """Group flat column names into ``{prefix: {field: index}}``.

    ``right_qx`` becomes prefix ``right``, field ``qx``; a bare ``shoulder_pan`` becomes prefix
    ``""``. The longest suffix that is a known field wins, so ``left_wrist_roll`` is not
    mistaken for a hand called ``left_wrist`` with a field ``roll``.
    """
    known = set(POSE_FIELDS) | set(GRIP_FIELDS)
    out = {}
    for i, n in enumerate(names):
        low = str(n).lower()
        prefix, fld = "", low
        for cut in range(1, low.count("_") + 1):
            head, tail = low.rsplit("_", cut)[0], "_".join(low.rsplit("_", cut)[1:])
            if tail in known:
                prefix, fld = head, tail
                break
        out.setdefault(prefix, {})[fld] = i
    return out


def _pose_hands(groups, act):
    hands = []
    for prefix, fields in groups.items():
        if not all(f in fields for f in POSE_FIELDS):
            continue
        idx = [fields[f] for f in POSE_FIELDS]
        grip = next((act[:, fields[g]] for g in GRIP_FIELDS if g in fields), None)
        hands.append(Hand(name=prefix or "hand", pos=act[:, idx[:3]],
                          quat=normalise_quats(act[:, idx[3:]]), grip=grip, source="pose"))
    return hands


def _joint_hands(names, act, chain):
    """Hands recovered from joint angles by forward kinematics, if the names line up."""
    from scipy.spatial.transform import Rotation as R

    want = [j.name for j in chain.joints]
    if not all(want):
        return []
    low = [str(n).lower() for n in names]
    hands = []
    prefixes = sorted({n[:-len(w)] for n in low for w in want if n.endswith(w)})
    for prefix in prefixes:
        idx = [low.index(prefix + w) for w in want if prefix + w in low]
        if len(idx) != len(want):
            continue
        q = act[:, idx]
        pos, M = chain.fk(q)
        grip = next((act[:, low.index(prefix + g)] for g in GRIP_FIELDS
                     if prefix + g in low), None)
        hands.append(Hand(name=(prefix.rstrip("_") or chain.name), pos=pos,
                          quat=R.from_matrix(M).as_quat(), grip=grip, source="fk", joints=q))
    return hands


def load(path, episode=None, chain=None, column="action"):
    """One episode of a LeRobot dataset, as ``pos``/``quat`` per hand.

    Args:
        path: the dataset directory (with ``meta/info.json``), or a single parquet file.
        episode: which episode; None takes the one with the most frames present.
        chain: needed only if the dataset stores joint angles rather than poses -- then the
            angles are run through its forward kinematics.
        column: ``"action"`` (what was commanded) or ``"observation.state"`` (what was
            measured). Commands are usually the cleaner signal to retarget.

    Returns:
        :class:`Episode`.

    Raises:
        ValueError: if nothing in the column looks like a pose or a known joint set. The message
            lists what the names actually were, because that is the thing you need to see.
    """
    root = Path(path)
    info = _info(root) if root.is_dir() else {}
    episode, act = _frame(root, column, episode)

    names = ((info.get("features") or {}).get(column) or {}).get("names")
    if isinstance(names, dict):                       # some writers nest names under the key
        names = next(iter(names.values()), None)
    if not names:
        names = [f"c{i}" for i in range(act.shape[1])]

    hands = _pose_hands(_split(names), act)
    if not hands and chain is not None:
        hands = _joint_hands(names, act, chain)
    if not hands:
        raise ValueError(
            f"nothing in {column!r} of {root} looks like an end-effector pose. Its columns are "
            f"{list(names)}.\nOmniBase needs position and orientation: name them "
            f"<hand>_x,_y,_z,_qx,_qy,_qz,_qw. If they are joint angles instead, pass "
            f"chain=... and they will be run through forward kinematics; if they are neither, "
            f"build pos (N, 3) and quat (N, 4) yourself and skip this loader.")

    return Episode(index=episode, hands=hands, fps=info.get("fps"),
                   version=info.get("codebase_version"), frames=len(act))


def describe_dataset(path, column="action", chain=None):
    """What is in a dataset and whether OmniBase can use it -- print this first."""
    root = Path(path)
    info = _info(root)
    lines = [f"{root}",
             f"  codebase {info.get('codebase_version', '?')}  fps {info.get('fps', '?')}  "
             f"episodes {info.get('total_episodes', '?')}"]
    feats = info.get("features") or {}
    for key in (column, "observation.state"):
        f = feats.get(key)
        if f:
            lines.append(f"  {key:20} shape {f.get('shape')}  names {f.get('names')}")
    try:
        ep = load(root, chain=chain, column=column)
        for h in ep.hands:
            lines.append(f"  -> hand {h.name!r}: {len(h.pos)} frames, {h.source}"
                         + (", gripper column found" if h.grip is not None else ""))
    except Exception as exc:                                # noqa: BLE001 -- reported, not raised
        lines.append(f"  -> unusable: {exc}")
    return "\n".join(lines)


def from_parquet(path, episode=None, hands=(0, 1), width=8, column="action",
                 episode_column="episode_index"):
    """Positional reader, for data with no ``meta/info.json`` to name the columns.

    Assumes each hand is a contiguous ``[x, y, z, qx, qy, qz, qw, grip]`` block. Prefer
    :func:`load`, which reads the names instead of counting.
    """
    root = Path(path)
    episode, act = _frame(root, column, episode)
    need = (max(hands) + 1) * width
    if act.shape[1] < need:
        raise ValueError(f"{column} is {act.shape[1]} wide; hands={hands} at width {width} "
                         f"needs {need}. Pass hands=/width= to match your layout.")
    out = []
    for h in hands:
        c = h * width
        grip = act[:, c + 7] if width > 7 else np.zeros(len(act))
        out.append((act[:, c:c + 3], normalise_quats(act[:, c + 3:c + 7]), grip))
    del episode_column
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
            f"quaternions are not unit length (worst |q| = {norm.max():.4f}, at row "
            f"{int(np.abs(norm - 1.0).argmax())} of {len(q)}). Check the column offset, and "
            "check the order really is (x, y, z, w) and not (w, x, y, z).")
    return q / norm[..., None]
