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
from functools import lru_cache
from pathlib import Path

import numpy as np

from .robots import SO101_APPROACH, SO101_BODY_IN_CHAIN

#: Which way a gripper reaches, in this library's body frame. See the robot it came from.
APPROACH = np.asarray(SO101_APPROACH, dtype=float)
BODY = np.asarray(SO101_BODY_IN_CHAIN, dtype=float)   # the gripper body's frame, in the chain's terminal frame
FINGERS = BODY @ APPROACH                             # the same direction in the body's own frame: +y

#: The seven fields that make a pose, in the order this library wants them.
POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
#: Names a gripper column might go by, lowercased.
GRIP_FIELDS = ("grip", "gripper", "jaw", "grasp", "width")
#: The other way a rig writes an orientation. Read with :data:`DEFAULT_EULER` unless told.
EULER_FIELDS = ("roll", "pitch", "yaw")
#: scipy's spelling of roll-pitch-yaw: extrinsic x, then y, then z. Lowercase is extrinsic;
#: a rig that wrote intrinsic angles wants ``"ZYX"``. ``fit_level`` tries both.
DEFAULT_EULER = "xyz"

#: Recording rigs whose frames are not this library's. Each entry says how to get from what the
#: rig wrote to (world z up, tool point between the jaws along the body's -y, as SO101_TOOL is).
#:
#: ``euler`` is how to read its three angles (scipy's spelling; lowercase extrinsic). ``grip``
#: is the closed and open value of its gripper column. ``tcp`` is a starting guess at how far
#: ahead of what it tracked the fingertips are.
#:
#: What is deliberately NOT here is which way the gripper points in its own frame. It was, as a
#: guess, and the guess was wrong: a FastUMI rig carries its tracker rotated, so the fingers run
#: along the tracker's +x rather than the -z its datasheet would suggest. ``omnibase level``
#: measures it from the recording instead, which is both more reliable and one less thing to be
#: quietly wrong about on a rig nobody here has seen.
FRAMES = {
    "fastumi": dict(euler="xyz", tcp=0.06, grip=(0.04, 1.0)),
}


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
    files = list(_files(root))
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


@lru_cache(maxsize=8)
def _files(root: Path):
    """The dataset's parquet files, listed once.

    A v2.1 dataset has one file per episode, and FastUMI's tasks have four thousand of them.
    Globbing that per episode -- which is what this did -- costs more than reading the data.
    """
    return tuple(sorted(root.glob("data/**/*.parquet"))) if root.is_dir() else (root,)


def _frame(root: Path, column: str, episode):
    """Every frame of one episode, whichever on-disk layout the dataset uses."""
    import pandas as pd

    files = list(_files(root))
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

    # Euler angles, second and only where they can mean one. ``roll`` is both half of a
    # roll-pitch-yaw orientation and the name of a joint, and reading it as a field cost the
    # first version of this a phantom hand called ``left_wrist`` -- so a prefixed one is only
    # taken when that prefix already has a position to go with it.
    for name, i in list(out.get("", {}).items()):
        for f in EULER_FIELDS:
            stem = name[: -len(f) - 1]
            if name.endswith("_" + f) and set(POSE_FIELDS[:3]) <= set(out.get(stem, {})):
                out[stem][f] = out[""].pop(name)
                break
    return out


def _pose_hands(groups, act, euler=DEFAULT_EULER):
    """Hands stored as poses -- orientation as a quaternion, or as three Euler angles.

    Euler angles are three numbers and twelve conventions, and nothing in a dataset says which.
    ``euler`` is scipy's spelling; lowercase is extrinsic (``"xyz"`` is roll-pitch-yaw as most
    rigs mean it), uppercase intrinsic. Get it wrong and every orientation is subtly turned,
    which is why :func:`fit_level` measures it rather than assuming.
    """
    from scipy.spatial.transform import Rotation as R

    hands = []
    for prefix, fields in groups.items():
        pose = all(f in fields for f in POSE_FIELDS)
        rpy = all(f in fields for f in EULER_FIELDS) and all(f in fields for f in POSE_FIELDS[:3])
        if not (pose or rpy):
            continue
        idx = [fields[f] for f in POSE_FIELDS[:3]]
        if pose:
            quat = normalise_quats(act[:, [fields[f] for f in POSE_FIELDS[3:]]])
        else:
            quat = R.from_euler(euler, act[:, [fields[f] for f in EULER_FIELDS]]).as_quat()
        grip = next((act[:, fields[g]] for g in GRIP_FIELDS if g in fields), None)
        hands.append(Hand(name=prefix or "hand", pos=act[:, idx],
                          quat=quat, grip=grip, source="pose"))
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


def _axes(spec):
    """A signed permutation matrix from a spec like ``"x,-z,y"``.

    Row ``i`` is the axis the spec names for slot ``i``, so the matrix maps a vector written in
    the spec's own axes into the frame those slots belong to. A spec that is a reflection rather
    than a rotation is refused: mirroring a recording is a bug that looks like a left hand.
    """
    rows = []
    for word in str(spec).split(","):
        word = word.strip()
        sign = -1.0 if word.startswith("-") else 1.0
        axis = word.lstrip("+-").lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis spec {spec!r}: {word!r} is not one of x, y, z (optionally -)")
        rows.append(sign * np.eye(3)[("x", "y", "z").index(axis)])
    A = np.array(rows, dtype=float)
    if A.shape != (3, 3) or abs(np.linalg.det(A) - 1.0) > 1e-9:
        raise ValueError(f"axis spec {spec!r} is not a rotation (three axes, right-handed)")
    return A


def _upright(up):
    """The shortest rotation taking ``up`` to +z -- what levels a recording onto its table."""
    up = np.asarray(up, dtype=float)
    up = up / max(np.linalg.norm(up), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _recipe(frame=None, level=None, **over):
    """Merge a built-in rig preset, a fitted scene calibration and explicit overrides."""
    r = dict(euler=DEFAULT_EULER, body=None, body_R=None, tcp=0.0, grip=None, up=None,
             lift=0.0)
    if isinstance(level, (str, Path)):
        level = json.loads(Path(level).read_text(encoding="utf-8"))
    sources = [FRAMES.get(frame) if frame else None]
    if level:
        sources += [FRAMES.get(level.get("frame")), level]
    sources.append({k: v for k, v in over.items() if v is not None})
    for src in sources:
        if src:
            r.update({k: v for k, v in src.items() if k in r and v is not None})
    return r


def _turned(hand):
    """The same hand with its orientation in the chain's terminal frame."""
    from scipy.spatial.transform import Rotation as R

    import dataclasses

    return dataclasses.replace(hand, quat=(R.from_quat(hand.quat) * R.from_matrix(BODY)).as_quat())


def _apply_frame(hand, r):
    """Put one hand in this library's frame: rig body axes, tool point, table level.

    The levelling is per episode, on purpose. A rig docked in the same slot every take shares
    its ORIENTATION across episodes -- measurably: the jaws point the same way at every grasp --
    but not its origin, which drifts by centimetres between sessions and tables. So ``up`` is a
    constant of the task and the height is read from this episode's own grasps: the frames where
    the jaw closed, which is the hand touching something that was standing on the table.
    """
    from scipy.spatial.transform import Rotation as R

    M = R.from_quat(hand.quat).as_matrix()
    B = np.asarray(r["body_R"], dtype=float) if r.get("body_R") is not None else (
        _axes(r["body"]) if r["body"] else None)
    if B is not None:
        M = M @ B.T                                    # the rig's body axes -> the gripper body's
    pos = np.asarray(hand.pos, dtype=float)
    if r["tcp"]:
        pos = pos + M @ (float(r["tcp"]) * FINGERS)    # tracked point -> out at the fingers
    if r["up"] is not None:
        L = _upright(r["up"])
        pos, M = pos @ L.T, np.einsum("ij,njk->nik", L, M)
        pos = pos - np.concatenate([pos[:, :2].mean(0), [table_height(pos, hand.grip)]])
        pos[:, 2] += float(r.get("lift") or 0.0)
    hand.pos, hand.quat = pos, R.from_matrix(M).as_quat()
    return hand


def table_height(pos, grip, low=10.0):
    """Where the surface is, in an already-levelled episode: under its lowest grasps.

    A grasp happens at table height plus half an object, a release can be into a box or onto a
    shelf, and the hand spends the rest of the episode in the air. The low percentile of the
    contact heights is the closest thing to the surface the recording contains.
    """
    pos = np.asarray(pos, dtype=float)
    if grip is None:
        return float(pos[:, 2].min())
    ev = contacts(grip)
    return float(np.percentile(pos[ev, 2], low)) if len(ev) else float(pos[:, 2].min())


def contacts(grip, lo=None, hi=None, closing=None):
    """Frames where the jaw changed state: every grasp and every release.

    These are the only frames at which a hand-held recording says anything about the world.
    Everything else is the hand in mid-air; a grasp is the hand touching an object that is
    standing on something, which is what makes them the points to fit a table to.

    ``closing`` narrows it to one kind. Both belong to the surface, so both are used to find it,
    but they are opposite events -- the hand arrives at a grasp and leaves after a release --
    and anything that asks which way the gripper was pointing or moving wants grasps alone.
    Mixing them put the first version of this check at 101 degrees, halfway between two answers
    that were each nearly right.
    """
    g = np.asarray(grip, dtype=float)
    lo = float(np.percentile(g, 5)) if lo is None else float(lo)
    hi = float(np.percentile(g, 95)) if hi is None else float(hi)
    if hi - lo < 1e-6:
        return np.zeros(0, dtype=int)
    closed = g < 0.5 * (lo + hi)
    if closing is None:
        return np.flatnonzero(closed[:-1] != closed[1:]) + 1
    if closing:
        return np.flatnonzero(~closed[:-1] & closed[1:]) + 1
    return np.flatnonzero(closed[:-1] & ~closed[1:]) + 1


def load(path, episode=None, chain=None, column="action", frame=None, level=None,
         tcp=None, euler=None, body=None, stride=1, terminal=True):
    """One episode of a LeRobot dataset, as ``pos``/``quat`` per hand.

    Args:
        path: the dataset directory (with ``meta/info.json``), or a single parquet file.
        episode: which episode; None takes the one with the most frames present.
        chain: needed only if the dataset stores joint angles rather than poses -- then the
            angles are run through its forward kinematics.
        column: ``"action"`` (what was commanded) or ``"observation.state"`` (what was
            measured). Commands are usually the cleaner signal to retarget.
        frame: a key of :data:`FRAMES` -- a rig whose body axes, Euler convention and tool
            offset are known, so the poses arrive in this library's frame instead of the rig's.
        level: a fitted scene calibration -- the path to the JSON ``omnibase level`` writes, or
            the dict itself. It carries the rest of the recipe too, so once fitted it is the
            only flag you need.
        tcp, euler, body: override one part of that recipe. ``tcp`` is metres from whatever the
            rig tracked to the point between the fingers.
        stride: keep every n-th frame. Retarget at the rate the policy will run at rather than
            the rate the rig recorded at, so a window means the same thing in both.

    Returns:
        :class:`Episode`.

    Raises:
        ValueError: if nothing in the column looks like a pose or a known joint set. The message
            lists what the names actually were, because that is the thing you need to see.
    """
    root = Path(path)
    info = _info(root) if root.is_dir() else {}
    episode, act = _frame(root, column, episode)
    r = _recipe(frame, level, tcp=tcp, euler=euler, body=body)
    if int(stride) > 1:
        act = act[::int(stride)]

    names = ((info.get("features") or {}).get(column) or {}).get("names")
    if isinstance(names, dict):                       # some writers nest names under the key
        names = next(iter(names.values()), None)
    if not names:
        names = [f"c{i}" for i in range(act.shape[1])]

    hands = [_apply_frame(h, r) for h in _pose_hands(_split(names), act, r["euler"])]
    if terminal:
        # Everything above is in the gripper BODY's frame -- jaws along +y, housing top +z, the
        # frame a recording of the gripper reports and the level fit reasons in. The solver works
        # in the chain's terminal frame, which is that body turned by BODY. Callers that analyse
        # the recording rather than solve it (fit_level) ask for terminal=False.
        hands = [_turned(h) for h in hands]
    if not hands and chain is not None:
        hands = _joint_hands(names, act, chain)
    if not hands:
        raise ValueError(
            f"nothing in {column!r} of {root} looks like an end-effector pose. Its columns are "
            f"{list(names)}.\nOmniBase needs position and orientation: name them "
            f"<hand>_x,_y,_z,_qx,_qy,_qz,_qw. If they are joint angles instead, pass "
            f"chain=... and they will be run through forward kinematics; if they are neither, "
            f"build pos (N, 3) and quat (N, 4) yourself and skip this loader.")

    fps = info.get("fps")
    return Episode(index=episode, hands=hands, fps=(fps / int(stride) if fps else None),
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


# ---------------------------------------------------------------------------------------------
# Where is the table?
#
# A hand-held rig reports where it moved, not where it was. FastUMI's poses start at exactly
# zero every episode, because they are relative to the tracker's own pose at frame 0 -- so the
# recording knows nothing about gravity, the floor, or which way is up.
#
# What makes it recoverable is that frame 0 is not arbitrary. The gripper is docked in a fixed
# slot on the collection rig before every take, so every episode of a task shares one origin,
# and the table is a property of the TASK rather than of the episode. Pool the frames where the
# jaw opened or closed -- the hand touching something that is standing on the table -- across a
# few hundred episodes, and one plane fits all of them.
#
# That same fit settles the two other unknowns for free. The Euler convention and the distance
# from the tracker to the fingertips both move the contact points, and only the right pair
# collapses grasps made at different wrist angles onto a single plane. So they are searched
# rather than assumed.
# ---------------------------------------------------------------------------------------------


def _flatness(points, trim=0.2):
    """A shared normal for contact points that were each recorded in their own origin.

    Every episode is levelled against its own grasps, so a normal is judged by whether it makes
    contacts flat WITHIN an episode, not by whether episodes agree about height. Centre each
    episode's points, stack them, and the least-varying direction of what is left is the answer
    -- an eigenvector, not a search. Two passes, dropping the worst ``trim`` of points in
    between: a grasp off a shelf or the rim of a box is a real contact and a false floor.

    Returns ``(normal, rms residual in metres, share of points kept)``.
    """
    if not points:
        return None, np.inf, 0.0
    X = np.concatenate([P - P.mean(0) for P in points if len(P)]) if points else None
    if X is None or len(X) < 3:
        return None, np.inf, 0.0
    keep = np.ones(len(X), dtype=bool)
    n = None
    for _ in range(2):
        n = np.linalg.svd(X[keep] - X[keep].mean(0))[2][-1]
        d = np.abs(X @ n)
        if trim > 0:
            keep = d <= np.quantile(d, 1.0 - trim)
    res = float(np.sqrt((( X[keep] @ n) ** 2).mean()))
    return n, res, float(keep.mean())


def approach_axis(takes, up):
    """Which way the fingers point, in the rig's own frame -- from where they point at a grasp.

    A gripper closing on something that is standing on a surface points at that surface. So the
    body direction whose world image is ``-up`` at every grasp is the finger axis, and it is a
    one-line average: rotate ``-up`` back into each grasp's body frame and add them up.

    The length of that average is the number to watch. It is 1.0 only if the same body direction
    points at the table at every grasp in every episode -- which cannot happen unless the
    episodes really do share a frame, the angles really were read in the right order, and the
    fitted table really is the table. Nothing else in this fit tests all three at once.

    The obvious alternative -- take the direction the hand MOVES in just before a grasp -- reads
    better and does not work: hands do not only descend onto things. Over 180 grasps of shoes
    the pre-grasp motion averaged 116 degrees away from straight down, because half of them
    reach up and over the side of a box first. That fit returned an axis with 0.54 consistency
    and the wrong answer; this one returns 0.97 and the right one.
    """
    S, n = np.zeros(3), 0
    down = -np.asarray(up, dtype=float)
    for _, M, grip in takes:
        ev = contacts(grip, closing=True)
        if len(ev):
            S += np.einsum("nji,j->i", M[ev], down)
            n += len(ev)
    if n == 0:
        return None, 0.0
    return S / np.linalg.norm(S), float(np.linalg.norm(S) / n)


def _body_from(approach, upref):
    """A rotation taking the rig's body axes to ours: fingers along +y, housing-up along +z."""
    y = np.asarray(approach, dtype=float)
    y = y / np.linalg.norm(y)
    z = np.asarray(upref, dtype=float)
    z = z - y * (z @ y)
    if np.linalg.norm(z) < 1e-6:                       # degenerate reference; any roll will do
        z = np.eye(3)[int(np.argmin(np.abs(y)))]
        z = z - y * (z @ y)
    z = z / np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z])            # rows: our x, y, z in the rig's axes


def _up_from(pts, alls):
    """``(up, rms residual, kept)`` -- the surface those contact points lie on, sign resolved."""
    n, res, kept = _flatness(pts)
    if n is None:
        return None, np.inf, 0.0, 1.0

    def under(v):
        return float(np.mean([((P @ v - np.percentile(Q @ v, 10)) < -0.02).mean()
                              for P, Q in zip(alls, pts)]))

    # Which side is up? A surface has things resting on it and a hand moving above it; almost
    # nothing goes under. Asking the jaws instead reads well and is wrong the moment a task
    # grasps from the side, which taking a plate out of a rack does.
    below, other = under(n), under(-n)
    if other < below:
        n, below = -n, other
    return n, res, kept, below


def fit_level(path, subset=None, frame="fastumi", column="action", eulers=("xyz",),
              tcps=(0.0, 0.03, 0.06, 0.09, 0.12, 0.15), tol=0.04, sample=150, seed=0,
              body=None, report=None):
    """Work out what a hand-held recording's numbers mean, from the recording.

    Four things have to be known before an arm can be asked to follow one, and a FastUMI-style
    dataset states none of them: which way is up, where the fingertips are relative to whatever
    was tracked, which way the gripper points in its own frame, and in what order to read its
    three Euler angles. Each is measured here from the part of the data that can see it.

    * **up**, from the frames where the jaw changed state. Those are the hand touching things
      that were standing on a surface, so the direction that makes them flat -- within an
      episode, since the rig's origin drifts between takes -- is up. Positions only, so no
      convention can corrupt it, and the sign is whichever leaves less of the recording
      underground.
    * **the finger axis**, from the direction that points at that surface whenever the jaws
      close (:func:`approach_axis`).
    * **the tool offset**, from which offset makes grasps taken at different wrist angles land
      on the same surface. Weakly determined: objects are different heights, and that noise is
      larger than the effect. Treat it as a starting point and check your answer against it.
    * **the Euler convention**, by the same consistency the finger axis is measured with. Read
      three angles in the wrong order and each frame's body frame is turned differently, so no
      single direction points at the table at every grasp and the consistency collapses. The
      recording's own wrist video agrees independently -- fitting how the picture pans, tilts
      and rolls against each candidate's rotations puts ``xyz`` three times ahead of the next.

    Returns the calibration :func:`load` takes, with the evidence: ``flat`` (rms millimetres a
    contact sits off the surface), ``aligned`` (how sharply one body axis points at the table
    across every grasp -- the statistic that tests the whole story at once), ``below`` (share of
    frames ending up under the table) and ``approach_deg`` (how far the fingers are from the
    table at a grasp).
    """
    root = Path(path)
    eps = list(episodes(root)) if subset is None else list(subset)
    rng = np.random.default_rng(seed)
    if sample and len(eps) > sample:
        eps = sorted(rng.choice(eps, sample, replace=False).tolist())

    preset = FRAMES.get(frame, {})
    best, trouble = None, None
    for euler in eulers:
        from scipy.spatial.transform import Rotation as R
        takes = []
        for e in eps:
            try:
                ep = load(root, episode=e, column=column, euler=euler, terminal=False)
            except Exception as exc:                 # noqa: BLE001 -- one bad episode, not the run
                trouble = trouble or f"episode {e}: {type(exc).__name__}: {exc}"
                continue
            for h in ep.hands:
                if h.grip is not None and len(contacts(h.grip)) >= 2:
                    takes.append((h.pos, R.from_quat(h.quat).as_matrix(), h.grip))
        if not takes:
            continue
        # Up first, from the tracked points themselves: the tool offset needs a direction to go
        # along, and that direction needs a table to point at.
        up0, _, _, _ = _up_from([p[contacts(g)] for p, _, g in takes], [p for p, _, _ in takes])
        if up0 is None:
            continue
        axis, aligned = approach_axis(takes, up0)
        if axis is None:
            continue
        for tcp in tcps:
            tips = [p + M @ (float(tcp) * axis) for p, M, _ in takes]
            pts = [t[contacts(g)] for t, (_, _, g) in zip(tips, takes)]
            n, res, kept, below = _up_from(pts, tips)
            if n is None:
                continue
            J = np.concatenate([M[contacts(g, closing=True)] @ axis for _, M, g in takes])
            approach = float(np.median(np.degrees(np.arccos(np.clip(-(J @ n), -1.0, 1.0)))))
            # Consistency decides the convention, flatness the tool offset, and ``below`` is
            # only a gate -- it falls monotonically with the offset for a reason unrelated to
            # being right, so scoring it would just pick the longest offset on the list.
            score = aligned - 2.0 * res - 0.5 * max(below - 0.03, 0.0)
            if best is None or score > best["score"]:
                upref = np.mean(np.concatenate(
                    [M[contacts(g, closing=True)].transpose(0, 2, 1) @ n
                     for _, M, g in takes[:50]]), axis=0)
                best = dict(frame=frame, euler=euler,
                            body_R=[list(map(float, row)) for row in _body_from(axis, upref)],
                            approach=[float(v) for v in axis], tcp=float(tcp), lift=0.0,
                            up=[float(v) for v in n],
                            grip=list(preset.get("grip") or []) or None,
                            score=round(float(score), 5), flat=round(1000 * float(res), 1),
                            kept=round(float(kept), 3), aligned=round(float(aligned), 3),
                            below=round(float(below), 4),
                            contacts=int(sum(len(P) for P in pts)), episodes=len(takes),
                            approach_deg=round(approach, 1), dataset=str(root))
            if report:
                report(euler, tcp, res, below, approach, score)
    if best is None:
        raise ValueError(
            f"nothing to fit a table to in {root}. Every episode has to load and have a gripper "
            f"column that changes; {len(eps)} were tried against column {column!r}."
            + (f"\nThe first one failed like this -- {trouble}" if trouble else
               "\nThey loaded, but no episode's gripper column ever changed state, so there is "
               "no moment of contact to fit to."))
    return best
