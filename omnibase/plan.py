"""Where would the robot have to stand?

A person demonstrating a task with a hand-held gripper does not think about a robot's reach.
Replay what they flew against a fixed-base arm and much of it is simply unreachable -- not
because the arm is slow or the controller is bad, but because no joint angles exist.

The move this library makes is to stop treating the base pose as a property of the robot. In a
recording it is a free variable: nothing in the data says where the robot was standing, because
there was no robot. So choose it -- and choose it again whenever it stops working.

That second part is what makes it pay. A policy never sees a whole episode at once; it sees an
observation window and predicts an action chunk, a couple of dozen frames. A base only has to
be right for that long. Cutting the episode into runs, each with its own placement, turns
"where can one base serve this whole demonstration" (often: nowhere) into "where can a base
serve the next twenty frames" (almost always: a wide region).

Two rules keep the result honest rather than merely large:

* the base is held until it genuinely stops working, never moved for its own sake, and when it
  must move it goes to the placement that will last LONGEST from here. Choosing the *nearest*
  workable placement instead makes it creep a grid cell at a time and shatters an episode into
  dozens of one-frame chunks that buy nothing.
* every arm in the scene shares one set of chunk boundaries. They are in one recording, and a
  cut that happens mid-reach for the other hand is not a cut you can use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Chunk:
    """A contiguous run of frames and the base placement each arm holds across it."""

    start: int
    stop: int
    bases: list                       # one (3,) placement per arm
    held: list = field(default_factory=list)   # fraction of frames each arm can hold
    #: Whether each arm is standing where it really stands, rather than somewhere invented.
    at_home: list = field(default_factory=list)

    def __len__(self):
        return self.stop - self.start


#: Rows per inverse-kinematics call, and (frame, cell) pairs per pruning pass. Both are only
#: memory ceilings: the solve is row-independent and the prune is pair-independent, so neither
#: changes an answer. They keep a long episode on a fine grid from asking for the whole product
#: at once -- 5000 frames x 1764 placements is 200 MB of target positions alone, per worker.
_IK_BATCH = 150_000
_PAIR_BATCH = 4_000_000


def base_grid(span=0.42, step=0.02, height=0.08, centre=(0.0, 0.0)):
    """Candidate base placements: a square lattice at one height.

    Height is a real lever and an easy one to forget -- low-and-close is the worst case for a
    small arm, and lifting the base a few centimetres can move more frames into reach than any
    amount of shuffling in the plane. Sweep it by calling this at several heights.
    """
    g = np.round(np.arange(-span, span + 1e-9, step), 6)
    return np.array([(centre[0] + x, centre[1] + y, height) for y in g for x in g])


def feasibility(chain, pos, quat, cells, pos_tol=0.015, rot_tol=np.radians(20.0)):
    """(frames, cells) boolean: could a base at this cell hold this pose?

    Solved once, here, because everything downstream is set algebra on the answer. The
    orientations are converted a single time and reused for every cell -- moving the base only
    moves the target's position, never its heading.

    Most of the pairs never reach the solver. Roughly three quarters of any real sweep is settled
    by arithmetic first -- the target is outside the arm's reach envelope, or held at an angle
    the wrist cannot make -- and :meth:`~omnibase.chain.Chain.cannot_reach` rejects those in
    bulk. What survives is gathered into one solve rather than one per cell, so a placement with
    four live frames does not cost what a placement with four hundred does. Neither step changes
    the answer: the prune only ever rejects what is provably out of reach, and the solve treats
    every row independently.
    """
    Rt = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    pos = np.asarray(pos, dtype=float)
    cells = np.asarray(cells, dtype=float)
    F = np.zeros((len(pos), len(cells)), dtype=bool)
    rows = max(1, _PAIR_BATCH // max(len(cells), 1))        # bound the (frames, cells, 3) block
    for f0 in range(0, len(pos), rows):
        f1 = min(f0 + rows, len(pos))
        rel = pos[f0:f1, None, :] - cells[None, :, :]  # a 4-column mount cell still fails loudly
        f, c = np.nonzero(~chain.cannot_reach(rel, Rt[f0:f1, None], pos_tol, rot_tol))
        for lo in range(0, f.size, _IK_BATCH):
            fi, ci = f[lo:lo + _IK_BATCH], c[lo:lo + _IK_BATCH]
            _, pe, re = chain.ik(rel[fi, ci], Rt[f0 + fi])
            F[f0 + fi, ci] = (pe < pos_tol) & (re < rot_tol)
    return F


def solve(chain, pos, quat, base, pos_tol=0.015, rot_tol=np.radians(20.0),
          smooth=np.radians(60.0)):
    """Joint angles for ONE base placement: ``(q, ok)``.

    :func:`feasibility` throws the angles away -- it answers a yes/no question over thousands of
    placements at once, and keeping ``q`` for every one of them is gigabytes. Once a placement
    has been chosen, ask again here and keep what comes back. Same solver, same tolerances, so
    ``ok`` is exactly the column :func:`feasibility` computed for that cell.

    One thing is added: continuity. The solver treats every frame as a fresh problem, which is
    right for counting what an arm can reach and wrong for writing down what it should do. A
    redundant joint has several ways to make the same pose, and picking each frame's
    independently lets it jump between them -- measured on FastUMI, a 320 degree step in
    wrist_roll between one frame and the next, in an episode where nothing else moved more than
    fifteen. That is not a bad base, it is the same base described twice; but a policy trained
    on it would learn to flick its wrist for no reason.

    So where a step exceeds ``smooth``, that one frame is solved again starting from the
    previous frame's answer, and the new answer is kept only if it still meets both tolerances
    and actually moves less. Nothing is smoothed, nothing is interpolated: every frame returned
    is a real solution to its own pose, and a genuinely large motion stays large.
    """
    Rt = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    rel = np.asarray(pos, dtype=float) - np.asarray(base, dtype=float)[:3]
    q, pe, re = chain.ik(rel, Rt)
    ok = (pe < pos_tol) & (re < rot_tol)
    if smooth:
        free = chain.free_joints()
        for i in range(1, len(q)):
            if not (ok[i] and ok[i - 1]):
                continue
            if np.abs(q[i] - q[i - 1]).max() <= smooth:
                continue
            # A free joint -- a wrist roll about the tool axis -- is swept for the best aim,
            # independently every frame, so two nearly-tied rolls can swap between neighbours.
            # It moves the tool point nowhere, so any value that still aims well enough is a
            # legitimate answer to this frame: take the one nearest where the wrist already was.
            for j in free:
                grid = np.linspace(chain.limits[j, 0], chain.limits[j, 1], 145)
                trial = np.tile(q[i], (len(grid), 1))
                trial[:, j] = grid
                _, re_g = chain.error(trial, np.tile(rel[i], (len(grid), 1)),
                                      np.tile(Rt[i], (len(grid), 1, 1)))
                good = np.flatnonzero(re_g < rot_tol)
                if good.size:
                    q[i, j] = grid[good[np.argmin(np.abs(grid[good] - q[i - 1, j]))]]
            # Anything still jumping is a different arm posture rather than a different wrist:
            # re-solve that one frame from where the arm already is, and keep the answer only
            # if it holds both tolerances and actually moves less.
            jump = np.abs(q[i] - q[i - 1]).max()
            if jump > smooth:
                qi, pi, ri = chain.ik(rel[i:i + 1], Rt[i:i + 1], seeds=[q[i - 1]])
                if pi[0] < pos_tol and ri[0] < rot_tol and np.abs(qi[0] - q[i - 1]).max() < jump:
                    q[i] = qi[0]
    return q, ok


def home_index(cells, home):
    """Which candidate placement is the real robot's own base.

    Accepts an index, or a coordinate that is snapped to the nearest candidate -- a real
    mounting is never exactly on your grid, and refusing it over a millimetre would be silly.
    """
    if home is None:
        return None
    cells = np.asarray(cells, dtype=float)
    if np.isscalar(home):
        return int(home)
    h = np.asarray(home, dtype=float).ravel()
    d = np.linalg.norm(cells[:, :3] - h[:3], axis=1)
    if cells.shape[1] > 3 and h.size > 3:            # match the mount heading too, if given
        d = d + np.abs(np.arctan2(np.sin(cells[:, 3] - h[3]), np.cos(cells[:, 3] - h[3])))
    return int(d.argmin())


def _runs(F):
    """``run[f, c]`` = first frame at or after f that cell c cannot hold. One backward sweep."""
    n, m = F.shape
    run = np.empty((n + 1, m), dtype=np.int32)
    run[n] = n
    for f in range(n - 1, -1, -1):
        run[f] = np.where(F[f], run[f + 1], f)
    return run


def chunk(Fs, cells, window=21, slack=None, home=None):
    """Cut the episode into runs, one base placement per arm per run.

    Args:
        Fs: one (frames, cells) feasibility mask per arm, from :func:`feasibility`.
        cells: the candidate placements those masks were computed over.
        window: how many frames one base must cover. Set this to what your policy actually
            consumes -- its observation buffer plus its action chunk -- not to the episode
            length. It is the whole lever.
        slack: when several placements last nearly as long as the best, prefer the one nearest
            where we already are. Defaults to ``window // 2`` frames of slack.
        home: where the robot actually stands -- an index into ``cells``, or a coordinate that
            gets snapped to the nearest one, or one per arm. Given it, the real base is used
            wherever it works and departed from only where it does not, returning as soon as it
            can. This is usually what you want: your robot has a base, and frames retargeted to
            it need no explanation at deployment, while frames retargeted somewhere invented are
            a claim about a robot standing where yours does not. ``Chunk.at_home`` says which is
            which, so you can weight or filter on it later.

    Returns:
        ``(chunks, feasible)`` -- a list of :class:`Chunk`, and an (arms, frames) mask of what
        is actually held under the placement each frame ends up assigned.

    Guaranteed: the fraction held is never worse than :func:`best_fixed`. Re-placing the base is
    a choice you may decline, so it cannot cost you anything, and a chunker that scored below
    standing still would be reporting its own bookkeeping rather than the data.
    """
    Fs = [np.asarray(F, dtype=bool) for F in Fs]
    cells = np.asarray(cells, dtype=float)
    n = Fs[0].shape[0]
    slack = window // 2 if slack is None else slack
    runs = [_runs(F) for F in Fs]
    homes = home if isinstance(home, (list, tuple)) and len(home) == len(Fs) else [home] * len(Fs)
    homes = [home_index(cells, h) for h in homes]
    # Where home could serve a whole window, per frame. Used to cut a displaced run short the
    # moment the real base becomes usable again -- without this, home is only reconsidered when
    # the stand-in fails, and the robot stays parked somewhere invented long after it could
    # have gone back.
    home_ok = [None if h is None else ((runs[a][np.arange(n), h] - np.arange(n))
                                       >= np.minimum(window, n - np.arange(n)))
               for a, h in enumerate(homes)]

    def choose(F, run, f, cur, home_i):
        """The cell to stand on from frame f, and the frame it stops working."""
        span = min(window, n - f)
        reach = run[f] - f
        ok = reach >= span
        # The real base first, whenever it can do the job. Not the longest-lasting placement,
        # not the nearest one -- a robot that already exists standing where it already stands.
        if home_i is not None and ok[home_i]:
            return home_i, int(run[f][home_i])
        if not ok.any():
            # Nothing can hold the whole window. Take the placement that holds the MOST of it,
            # not the one with the longest unbroken run -- those are different cells, and
            # picking by run length made chunking score below a single fixed base on data where
            # no placement ever covers a window. Chunking is a free choice on top of standing
            # still; it must never come out worse than standing still.
            cover = F[f:f + span].sum(axis=0)
            # If nothing holds a single frame here, stay where we are rather than wander to an
            # arbitrary cell: moving buys nothing and a plan that jumps for no reason is one
            # more thing for a reader to distrust.
            nxt = cur if (cur is not None and cover.max() == 0) else int(cover.argmax())
            return nxt, f + span
        if cur is not None and ok[cur]:
            return cur, int(run[f][cur])       # still working: never move for its own sake
        tied = np.where(ok & (reach >= reach.max(initial=0) - slack))[0]
        ref = cells[cur][:2] if cur is not None else np.zeros(2)
        nxt = int(tied[np.argmin(np.linalg.norm(cells[tied][:, :2] - ref, axis=1))])
        return nxt, int(run[f][nxt])

    chunks, f, cur = [], 0, [None] * len(Fs)
    while f < n:
        picks, stops = [], []
        for a, run in enumerate(runs):
            c, stop = choose(Fs[a], run, f, cur[a], homes[a])
            if homes[a] is not None and c != homes[a]:
                # Standing somewhere invented: come back as soon as home can take over.
                back = np.flatnonzero(home_ok[a][f + 1:stop])
                if back.size:
                    stop = f + 1 + int(back[0])
            picks.append(c)
            stops.append(stop)
        stop = max(f + 1, min(stops))          # the run ends as soon as ANY arm must move
        if picks == cur and chunks:            # nobody moved: extend rather than split
            chunks[-1].stop = stop
        else:
            chunks.append(Chunk(f, stop, [cells[c] for c in picks]))
        cur, f = picks, stop

    feas = np.zeros((len(Fs), n), dtype=bool)
    for ch in chunks:
        ch.held, ch.at_home = [], []
        for a, b in enumerate(ch.bases):
            c = int(np.where((cells == b).all(1))[0][0])
            feas[a, ch.start:ch.stop] = Fs[a][ch.start:ch.stop, c]
            ch.held.append(float(feas[a, ch.start:ch.stop].mean()))
            ch.at_home.append(homes[a] is not None and c == homes[a])
    return chunks, feas


def home_share(chunks, frames=None):
    """Fraction of frames that were served from the robot's own base, per arm.

    The number to watch when you have a real robot. Frames at home are directly deployable --
    same geometry, nothing to justify. Frames elsewhere are still useful training data, but they
    describe a robot standing where yours does not, and it is worth knowing how much of your set
    that is before you train on it.
    """
    if not chunks:
        return []
    arms = len(chunks[0].bases)
    total = frames or sum(len(c) for c in chunks)
    return [sum(len(c) for c in chunks if c.at_home and c.at_home[a]) / max(total, 1)
            for a in range(arms)]


def best_fixed(F, cells):
    """The single best placement for the whole recording: ``(cell, fraction held)``.

    The honest thing to compare against. If one base already serves the episode, chunking has
    bought nothing and should not be claimed to.
    """
    share = F.mean(0)
    i = int(share.argmax())
    return cells[i], float(share[i])


def yield_curve(F, cells, windows=(1, 21, 50, 100, 250, None)):
    """How much of a recording survives, as a function of how long one base must serve.

    ``None`` means the whole episode. This is the curve that says whether the idea is worth
    anything on YOUR data: if it is flat, one base was always fine; if it climbs steeply as the
    window shortens, re-placing the base is buying you most of your dataset back.
    """
    n = F.shape[0]
    out = []
    for w in windows:
        w_eff = n if w is None else w
        run = _runs(F)
        usable, mult = [], []
        for f in range(0, n, max(1, w_eff)):
            span = min(w_eff, n - f)
            ok = (run[f] - f) >= span
            usable.append(bool(ok.any()))
            mult.append(int(ok.sum()))
        out.append(dict(window=w, usable=float(np.mean(usable)),
                        median_placements=float(np.median(mult)),
                        max_placements=int(np.max(mult)) if mult else 0))
    return out


# ---------------------------------------------------------------------------------------------
# Where should the robot stand?
#
# Reachability alone is a poor answer. A base can reach every frame and still be a bad place to
# put a robot: if the arm is stretched flat, or riding a joint stop, or passing through a
# singularity, the demonstrations you generate there are ones a policy will struggle to imitate
# and a real arm will struggle to execute. Those are the placements that quietly produce bad
# data. So score them on all of it, and say which is which.
# ---------------------------------------------------------------------------------------------

#: What each component means, and which way is better. Kept next to the code because a score
#: nobody can interpret is just a number.
SCORE_TERMS = {
    "coverage": "fraction of frames the arm can hold at all -- the thing that must be high",
    "limit_margin": "how far the joints stay from their stops, 1 = mid-range, 0 = on the stop",
    "manipulability": "distance from singularity; low means the arm is stretched flat or "
                      "folded, where small pose changes need huge joint changes",
    "rot_margin": "how much of the orientation tolerance is left unspent",
}


def score_map(chain, pos, quat, cells, pos_tol=0.015, rot_tol=np.radians(20.0),
              weights=(0.55, 0.20, 0.15, 0.10)):
    """Score every candidate base placement, not just pass/fail.

    Args:
        weights: how much coverage, limit margin, manipulability and orientation margin count
            toward the composite. Coverage dominates by default -- a comfortable placement that
            cannot reach the task is worth nothing -- but the rest decide between placements
            that all reach.

    Returns:
        dict of (cells,) arrays: ``coverage``, ``limit_margin``, ``manipulability``,
        ``rot_margin``, ``score`` (0-1, higher better), and ``verdict``, one of ``"good"``,
        ``"workable"``, ``"marginal"`` or ``"unusable"``.

    Every component is computed from the joint angles actually solved for, so a placement that
    only reaches by jamming a joint against its stop scores badly even though it "reaches".
    """
    Rt = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    pos = np.asarray(pos, dtype=float)
    cells = np.asarray(cells, dtype=float)
    lo, hi = chain.limits[:, 0], chain.limits[:, 1]
    half = np.maximum((hi - lo) / 2.0, 1e-9)

    cov = np.zeros(len(cells))
    marg = np.zeros(len(cells))
    manip = np.zeros(len(cells))
    rotm = np.zeros(len(cells))
    for i, b in enumerate(cells):
        rel = pos - b
        live = ~chain.cannot_reach(rel, Rt, pos_tol, rot_tol)   # skip what arithmetic settles
        if not live.any():
            continue
        q, pe, re = chain.ik(rel[live], Rt[live])
        ok = (pe < pos_tol) & (re < rot_tol)
        cov[i] = ok.sum() / len(pos)
        if not ok.any():
            continue
        qk = q[ok]
        # Nearest stop, per frame, normalised so 1 is dead centre of the range.
        d = np.minimum(qk - lo, hi - qk) / half
        marg[i] = float(np.percentile(d.min(axis=1), 10))     # the tight frames, not the mean
        _, _, Jv, _ = chain.jacobian(qk)
        w = np.sqrt(np.maximum(np.linalg.det(Jv @ np.swapaxes(Jv, -1, -2)), 0.0))
        manip[i] = float(np.percentile(w, 10))
        rotm[i] = float(1.0 - np.clip(np.median(re[ok]) / rot_tol, 0.0, 1.0))

    manip_n = manip / manip.max() if manip.max() > 0 else manip
    wc, wl, wm, wr = weights
    score = wc * cov + wl * marg * (cov > 0) + wm * manip_n + wr * rotm * (cov > 0)
    score = np.clip(score / max(sum(weights), 1e-9), 0.0, 1.0)

    verdict = np.full(len(cells), "unusable", dtype=object)
    verdict[cov >= 0.50] = "marginal"
    verdict[(cov >= 0.85) & (marg >= 0.05)] = "workable"
    verdict[(cov >= 0.98) & (marg >= 0.12) & (manip_n >= 0.35)] = "good"
    return dict(cells=cells, coverage=cov, limit_margin=marg, manipulability=manip_n,
                rot_margin=rotm, score=score, verdict=verdict)


def best_spot(smap, require="good"):
    """The highest-scoring placement, preferring ones that earn a given verdict.

    Returns ``(cell, index, report)`` where ``report`` is a one-line human summary. If nothing
    earns the verdict asked for, it falls back to the best of whatever exists and says so.
    """
    order = np.argsort(smap["score"])[::-1]
    ranked = [i for i in order if smap["verdict"][i] == require] or list(order)
    i = int(ranked[0])
    note = "" if smap["verdict"][i] == require else f" (nothing rated {require!r})"
    report = (f"({smap['cells'][i][0]:+.3f}, {smap['cells'][i][1]:+.3f}, "
              f"{smap['cells'][i][2]:+.3f}) m  score {smap['score'][i]:.3f}  "
              f"[{smap['verdict'][i]}]  coverage {100 * smap['coverage'][i]:.0f}%, "
              f"limit margin {smap['limit_margin'][i]:.2f}, "
              f"manipulability {smap['manipulability'][i]:.2f}{note}")
    return smap["cells"][i], i, report


def ascii_map(smap, key="score", width=None):
    """The map as text, because a placement you cannot see is a number you cannot check.

    Densest character is best. Cells that cannot reach the task at all are blank, so the shape
    of the reachable region is visible at a glance rather than inferred from a table.
    """
    cells = smap["cells"]
    xs = np.unique(cells[:, 0])
    ys = np.unique(cells[:, 1])
    v = smap[key]
    ramp = " .:-=+*#%@"
    grid = {(round(float(c[0]), 6), round(float(c[1]), 6)): i for i, c in enumerate(cells)}
    top = v.max() if v.max() > 0 else 1.0
    lines = [f"  {key}: blank = cannot reach, '{ramp[-1]}' = best ({top:.3f})",
             f"  x {xs.min():+.2f} .. {xs.max():+.2f} m down, y {ys.min():+.2f} .. "
             f"{ys.max():+.2f} m across"]
    step = max(1, len(ys) // (width or len(ys)))
    for x in xs[::-1]:
        row = []
        for y in ys[::step]:
            i = grid.get((round(float(x), 6), round(float(y), 6)))
            if i is None or smap["coverage"][i] <= 0:
                row.append(" ")
            else:
                row.append(ramp[min(len(ramp) - 1, int(v[i] / top * (len(ramp) - 1)))])
        lines.append(f"  {x:+.2f} |" + "".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Arms that cannot be placed separately
#
# Everything above places each arm on its own. That is right for arms on separate stands, and
# wrong for almost every real bimanual robot: two arms on one torso, a humanoid, a pair bolted
# to the same rail. There the spacing is a property of the hardware and the only thing you can
# choose is where the whole assembly stands -- and a placement that suits the right arm is no
# use if it puts the left one somewhere it cannot reach.
#
# So the assembly is placed instead of the arms. Each arm sits at a fixed offset from a mount
# frame, and a candidate is feasible only where EVERY arm can hold its own trajectory. The
# result is strictly worse than placing them independently, which is the honest point: it is
# what the hardware actually permits.
# ---------------------------------------------------------------------------------------------


@dataclass
class Mount:
    """Arms rigidly attached to one frame -- a torso, a rail, a bimanual base plate.

    Args:
        offsets: one ``(x, y, z)`` per arm, or ``(x, y, z, qx, qy, qz, qw)`` if an arm is also
            turned relative to the mount, which shoulders usually are.
        names: what to call each arm in reports.
    """

    offsets: list
    names: list = None

    def __post_init__(self):
        fixed = []
        for o in self.offsets:
            o = np.asarray(o, dtype=float)
            if o.shape == (3,):
                o = np.concatenate([o, [0.0, 0.0, 0.0, 1.0]])
            if o.shape != (7,):
                raise ValueError(f"a mount offset is (x,y,z) or (x,y,z,qx,qy,qz,qw), got {o.shape}")
            fixed.append(o)
        self.offsets = fixed
        if self.names is None:
            self.names = [f"arm{i}" for i in range(len(fixed))]

    def __len__(self):
        return len(self.offsets)


def pair(separation, along="y", names=("right", "left")):
    """The common case: two identical arms a fixed distance apart, facing the same way.

    ``separation`` is centre to centre, so ``pair(0.46)`` puts them at -0.23 and +0.23.
    """
    a = {"x": 0, "y": 1, "z": 2}[along]
    offs = []
    for sign in (-1.0, 1.0):
        o = np.zeros(3)
        o[a] = sign * separation / 2.0
        offs.append(o)
    return Mount(offs, list(names))


def mount_grid(span=0.42, step=0.02, height=0.08, yaws=(0.0,), centre=(0.0, 0.0)):
    """Candidate placements for a whole assembly: ``(x, y, z, yaw)`` rows.

    Yaw matters here in a way it does not for a single arm. Turning one arm about its own base
    is the same as turning the target around it, and a shoulder joint absorbs it; turning a
    torso swings the second arm through an arc, which changes what it can reach. Sweep a few.
    """
    g = np.round(np.arange(-span, span + 1e-9, step), 6)
    return np.array([(centre[0] + x, centre[1] + y, height, w)
                     for w in yaws for y in g for x in g])


def _yawmat(w):
    c, s = np.cos(w), np.sin(w)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_bases(mount, cell):
    """Where each arm's base ends up, for one assembly placement.

    Returns a list of ``(position, rotation matrix)`` -- what you would bolt down, or hand to a
    simulator as the arm's root pose.
    """
    cell = np.asarray(cell, dtype=float)
    t, yaw = cell[:3], (cell[3] if cell.size > 3 else 0.0)
    Y = _yawmat(yaw)
    out = []
    for off in mount.offsets:
        out.append((t + Y @ off[:3], Y @ R.from_quat(off[3:]).as_matrix()))
    return out


def mount_feasibility(chains, hands, mount, cells, pos_tol=0.015, rot_tol=np.radians(20.0)):
    """Where the whole assembly can stand: ``(combined, per_arm)`` masks over ``cells``.

    Args:
        chains: one :class:`~omnibase.chain.Chain` shared by every arm, or one per arm.
        hands: ``(pos, quat)`` per arm, in the same order as ``mount.offsets``.

    ``combined[f, c]`` is true only where every arm can hold frame ``f`` with the assembly at
    ``c``. Feed it straight to :func:`chunk` as a single mask and the chunks come back with one
    placement each -- the assembly's, which is the only thing there was to choose.
    """
    if not isinstance(chains, (list, tuple)):
        chains = [chains] * len(mount)
    if not (len(chains) == len(hands) == len(mount)):
        raise ValueError(f"{len(chains)} chain(s), {len(hands)} hand(s) and {len(mount)} mount "
                         "offset(s) must agree")
    cells = np.asarray(cells, dtype=float)
    n = len(hands[0][0])
    per = [np.zeros((n, len(cells)), dtype=bool) for _ in hands]

    # The mount's yaw turns the arm, so the targets have to be brought into the turned frame.
    # Doing it per distinct yaw rather than per cell keeps the rotation work off the inner loop.
    yaws = cells[:, 3] if cells.shape[1] > 3 else np.zeros(len(cells))
    for w in np.unique(yaws):
        Y = _yawmat(w)
        which = np.where(yaws == w)[0]
        for a, (pos, quat) in enumerate(hands):
            Rt = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
            off = mount.offsets[a]
            A = (Y @ R.from_quat(off[3:]).as_matrix())          # arm frame in the world
            p_in = np.asarray(pos, dtype=float) @ A             # A.T @ p, batched
            R_in = np.einsum("ij,njk->nik", A.T, Rt)
            for c in which:
                base = cells[c, :3] + Y @ off[:3]
                rel = p_in - base @ A
                live = ~chains[a].cannot_reach(rel, R_in, pos_tol, rot_tol)
                if not live.any():
                    continue
                _, pe, re = chains[a].ik(rel[live], R_in[live])
                per[a][live, c] = (pe < pos_tol) & (re < rot_tol)
    combined = np.logical_and.reduce(per)
    return combined, per
