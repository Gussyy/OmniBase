"""Evaluation without a simulator.

Two questions about a policy, or about a dataset before any policy exists, answered by
re-solving the demonstrations from other base placements and looking at what changes.

**Is base augmentation safe for this action space?** (:func:`ambiguity`) Writing one
demonstration from several bases gives an image-conditioned policy one observation with several
joint-space labels -- the wrist camera rides on the hand, so the picture is the same from every
base. It learns their mean. Push that mean through forward kinematics from each base and the
miss at the tool is in centimetres, before anything is trained. For absolute joint targets it
is about the spread of the bases; for joint deltas it is under a centimetre; for end-effector
actions it is zero by construction.

**Does a policy know where it stands, or has it memorised the motion?** (:func:`probe`) Move
the base by ``b``, re-solve the same frames, feed the policy the re-solved state, and put its
prediction through forward kinematics. The miss against the re-solved target is base-free and
in centimetres. Two references bracket every answer: replaying the unshifted rows misses by
exactly ``|b|`` (this is what calibrates the number), and the re-solved rows miss by nothing.
A policy that only ever saw one base lands on the first line; one that reads its state can
stay near the second. If the policy has a likelihood, the same contrast is available in nats.

What neither says: whether the policy will succeed. Measured on PushT with a public diffusion
policy, no offline score of a demonstration -- loss, sampled-chunk distance, spread, this
perturbation contrast -- separates the starts it succeeds from (AUC 0.48-0.57; chance is 0.5).
Success is decided on the policy's own states at a few critical frames, and a model's opinion
of a human's path says nothing about that. These are instruments for data questions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load, pick_hands
from .plan import solve
from .robots import load as load_robot


def chunks_from_plan(plan, chain=None, hands=None):
    """The planned chunks of a sweep, as hand trajectories: ``(chunks, chain, cfg)``.

    Each chunk is ``dict(pos, quat, grip, base, episode, start)``, the frames one base was
    chosen for and where it stood. ``plan`` is the file ``omnibase sweep --out`` wrote, or
    the loaded document.
    """
    doc = json.loads(Path(plan).read_text(encoding="utf-8")) if isinstance(plan, (str, Path)) else plan
    cfg = doc["cfg"]
    chain = chain or load_robot(cfg.get("robot", "so101"))
    out = []
    for rec in doc["episodes"]:
        if "error" in rec or not rec.get("chunks"):
            continue
        ep = load(cfg["dataset"], episode=rec["episode"], chain=chain, column=cfg["column"],
                  frame=cfg.get("frame"), level=cfg.get("level"), tcp=cfg.get("tcp"),
                  stride=cfg.get("stride", 1) or 1)
        for a, hand in enumerate(pick_hands(ep, hands or cfg.get("hands"))):
            for ch in rec["chunks"]:
                s0, s1 = int(ch["start"]), int(ch["stop"])
                out.append(dict(pos=hand.pos[s0:s1], quat=hand.quat[s0:s1],
                                grip=None if hand.grip is None else hand.grip[s0:s1],
                                base=np.asarray(ch["bases"][min(a, len(ch["bases"]) - 1)], float),
                                episode=int(rec["episode"]), hand=hand.name, start=s0))
    return out, chain, cfg


def resolve(chain, chunks, offset=(0.0, 0.0, 0.0), pos_tol=0.015, rot_tol=np.radians(20.0)):
    """Every chunk re-solved with its base moved by ``offset``: ``(state, action, ok, where)``.

    ``state[i]`` and ``action[i]`` are consecutive frames' joint angles (radians); ``ok[i]`` says
    both solved within tolerance; ``where[i]`` is ``(episode, frame)`` in the source dataset,
    for a policy that needs to look up the picture that goes with the state.
    """
    S, A, M, W = [], [], [], []
    for c in chunks:
        q, ok = solve(chain, c["pos"], c["quat"], c["base"] + np.asarray(offset, float), pos_tol, rot_tol)
        S.append(q[:-1]); A.append(q[1:]); M.append(ok[:-1] & ok[1:])
        W.append(np.stack([np.full(len(q) - 1, c["episode"]), c["start"] + np.arange(len(q) - 1)], 1))
    return np.concatenate(S), np.concatenate(A), np.concatenate(M), np.concatenate(W)


def grasp_window(chunks, rows=5):
    """Mask over the frame pairs of :func:`resolve`: within ``rows`` of the jaw first closing.

    Success is decided there, not in free space. A chunk that starts already closed has no
    grasp in it and contributes nothing.
    """
    out = []
    for c in chunks:
        n = len(c["pos"]) - 1
        w = np.zeros(n, dtype=bool)
        g = c["grip"]
        if g is not None and len(g) > 1:
            lo, hi = float(np.percentile(g, 5)), float(np.percentile(g, 95))
            closed = np.asarray(g, float) < 0.5 * (lo + hi) if hi - lo > 1e-6 else np.zeros(len(g), bool)
            if closed.any() and not closed[0]:
                t = int(np.argmax(closed))
                w[max(0, t - rows): t + rows] = True
        out.append(w)
    return np.concatenate(out)


def _offsets(cms):
    """``2,4`` -> the four compass directions at each radius, plus zero."""
    out = [("0", np.zeros(3))]
    for cm in cms:
        for tag, (dx, dy) in (("x+", (1, 0)), ("x-", (-1, 0)), ("y+", (0, 1)), ("y-", (0, -1))):
            out.append((f"{tag}{cm:g}", np.array([dx * cm / 100, dy * cm / 100, 0.0])))
    return out


# ---- is augmentation safe? ------------------------------------------------------------------

def ambiguity(chain, chunks, spans_cm=(2, 4, 6), zspan_cm=0.0, pos_tol=0.015, rot_tol=np.radians(20.0)):
    """The miss, in centimetres, of a policy that averages the joint actions of a base grid.

    For each half-width in ``spans_cm`` the demonstration is solved from a 3x3 grid of bases
    (3x3x3 with ``zspan_cm``); frames every base can hold are kept; the mean joint action is
    pushed through FK from each base and compared with that base's own target.

    Returns one row per span: ``dict(span_cm, bases, frames, absolute_cm, absolute_p90_cm,
    delta_cm, delta_p90_cm)``. ``absolute`` is absolute joint targets, ``delta`` is
    next-minus-current joints; end-effector actions would be zero and are not computed.
    """
    rows = []
    for span in spans_cm:
        s = span / 100.0
        zs = [0.0] if not zspan_cm else [-zspan_cm / 100.0, 0.0, zspan_cm / 100.0]
        offs = [np.array([x, y, z]) for x in (-s, 0, s) for y in (-s, 0, s) for z in zs] if span else [np.zeros(3)]
        miss_abs, miss_del, n = [], [], 0
        for c in chunks:
            sols = [solve(chain, c["pos"], c["quat"], c["base"] + o, pos_tol, rot_tol) for o in offs]
            ok = np.all([k[:-1] & k[1:] for _, k in sols], axis=0)
            if not ok.any():
                continue
            S = np.stack([q[:-1] for q, _ in sols]); A = np.stack([q[1:] for q, _ in sols])
            mean_abs, mean_del = A.mean(0), (A - S).mean(0)
            for bi in range(len(offs)):
                target = chain.fk(A[bi][ok])[0]
                miss_abs.append(np.linalg.norm(chain.fk(mean_abs[ok])[0] - target, axis=1))
                miss_del.append(np.linalg.norm(chain.fk(S[bi][ok] + mean_del[ok])[0] - target, axis=1))
            n += int(ok.sum())
        row = dict(span_cm=float(span), bases=len(offs), frames=n)
        if miss_abs:
            ma, md = 100 * np.concatenate(miss_abs), 100 * np.concatenate(miss_del)
            row.update(absolute_cm=float(np.median(ma)), absolute_p90_cm=float(np.percentile(ma, 90)),
                       delta_cm=float(np.median(md)), delta_p90_cm=float(np.percentile(md, 90)))
        rows.append(row)
    return rows


def ambiguity_table(rows):
    lines = ["| base spread (+- cm) | bases | frames held by all | absolute joints: miss cm (p90) | joint deltas: miss cm (p90) | end-effector |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if "absolute_cm" in r:
            lines.append(f"| {r['span_cm']:g} | {r['bases']} | {r['frames']} | {r['absolute_cm']:.1f} ({r['absolute_p90_cm']:.1f}) "
                         f"| {r['delta_cm']:.1f} ({r['delta_p90_cm']:.1f}) | 0 |")
        else:
            lines.append(f"| {r['span_cm']:g} | {r['bases']} | 0 | - | - | 0 |")
    return "\n".join(lines)


# ---- does the policy know where it stands? -------------------------------------------------

def knn_policy(state, action):
    """The simplest state-conditioned policy: return the action of the nearest training state.

    Enough to show the mechanism. Trained on one base it memorises; trained on a grid of bases
    it tracks the shift. It ignores ``where``, so it never looks at a picture.
    """
    S, A = np.asarray(state, float), np.asarray(action, float)

    def f(q, where=None):
        q = np.asarray(q, float)
        d = ((q[:, None, :] - S[None, :, :]) ** 2).sum(-1)
        return A[d.argmin(1)]
    f.__name__ = "knn"
    return f


def probe(chain, chunks, policies=None, offsets_cm=(2, 4, 6, 8), pos_tol=0.015, rot_tol=np.radians(20.0),
          grasp_rows=5, knn_grid_cm=None):
    """Miss at the tool point, in centimetres, against a shift of the base. No rollout.

    Args:
        policies: ``{name: policy}`` where ``policy(state, where) -> action``, joint angles in
            radians, ``where`` an ``(N, 2)`` array of ``(episode, frame)`` into the source
            dataset for policies that want the picture too. A policy may also carry
            ``policy.nll(state, action, where) -> (N,)`` in nats; if so the likelihood contrast
            is reported as well.
        knn_grid_cm: if given, two nearest-neighbour policies are added for reference: one fit
            at the plan's own bases and one fit on this grid of base offsets (e.g.
            ``(-6, -3, 0, 3, 6)``, applied to x and y). The pair shows what the probe looks
            like for a policy that memorised one base and for one that saw many.

    Returns ``dict(offsets=[tag...], rows={tag: {...}}, names=[...])``. Per offset: ``frames``,
    ``feasible`` (share of frames the re-solved base can hold), and per policy ``miss_cm``
    (median over all frames), ``grasp_cm`` (median within ``grasp_rows`` of the jaws closing,
    NaN if the chunks have no grasp), and ``nll`` / ``nll_replay`` if the policy has one.
    """
    policies = dict(policies or {})
    S0, A0, M0, W0 = resolve(chain, chunks, (0, 0, 0), pos_tol, rot_tol)
    G = grasp_window(chunks, grasp_rows)
    if knn_grid_cm is not None:
        policies["knn (one base)"] = knn_policy(S0[M0], A0[M0])
        Sg, Ag = [], []
        for gx in knn_grid_cm:
            for gy in knn_grid_cm:
                S, A, M, _ = resolve(chain, chunks, (gx / 100, gy / 100, 0), pos_tol, rot_tol)
                Sg.append(S[M]); Ag.append(A[M])
        policies["knn (base grid)"] = knn_policy(np.concatenate(Sg), np.concatenate(Ag))
    names = ["replay (unshifted rows)", "joint deltas (unshifted)"] + list(policies) + ["oracle (re-solved)"]

    def fk_cm(q):
        return 100 * chain.fk(q)[0]

    rows = {}
    for tag, b in _offsets(offsets_cm):
        Sb, Ab, Mb, Wb = resolve(chain, chunks, b, pos_tol, rot_tol)
        m = M0 & Mb
        if not m.any():
            rows[tag] = dict(frames=0, feasible=float(Mb.mean())); continue
        target = fk_cm(Ab[m]); g = G[m]
        preds = {"replay (unshifted rows)": A0[m], "joint deltas (unshifted)": Sb[m] + (A0[m] - S0[m])}
        preds.update({k: np.asarray(f(Sb[m], Wb[m]), float) for k, f in policies.items()})
        preds["oracle (re-solved)"] = Ab[m]
        row = dict(frames=int(m.sum()), feasible=float(Mb.mean()), grasp_frames=int(g.sum()))
        for k, v in preds.items():
            miss = np.linalg.norm(fk_cm(v) - target, axis=1)
            row[k] = dict(miss_cm=float(np.median(miss)), grasp_cm=float(np.median(miss[g])) if g.any() else float("nan"))
            nll = getattr(policies.get(k), "nll", None)
            if nll is not None:
                row[k]["nll"] = float(np.median(nll(Sb[m], Ab[m], Wb[m])))
                row[k]["nll_replay"] = float(np.median(nll(Sb[m], A0[m], Wb[m])))
        rows[tag] = row
    return dict(offsets=[t for t, _ in _offsets(offsets_cm)], names=names, rows=rows)


def probe_table(res, key="miss_cm"):
    """One line per policy, one column per shift radius (median over the four directions)."""
    tags = res["offsets"]; radii = ["0"] + sorted({t[2:] for t in tags if t != "0"}, key=float)

    def at(name, r):
        ts = ["0"] if r == "0" else [t for t in tags if t != "0" and t[2:] == r]
        v = [res["rows"][t][name][key] for t in ts if name in res["rows"][t]]
        return float(np.median(v)) if v else float("nan")

    unit = {"miss_cm": "cm", "grasp_cm": "cm, grasp rows", "nll": "nats", "nll_replay": "nats"}.get(key, "")
    lines = [f"| policy ({unit}) | " + " | ".join(f"{r} cm" for r in radii) + " |", "|---|" + "---|" * len(radii)]
    for n in res["names"]:
        if any(key in res["rows"][t].get(n, {}) for t in tags):
            lines.append(f"| {n} | " + " | ".join(f"{at(n, r):.1f}" for r in radii) + " |")
    feas = [float(np.median([res["rows"][t]["feasible"] for t in (["0"] if r == "0" else [t for t in tags if t != "0" and t[2:] == r])])) for r in radii]
    lines.append("| re-solve feasible | " + " | ".join(f"{100 * f:.0f}%" for f in feas) + " |")
    return "\n".join(lines)


def load_policy(spec):
    """``module:attribute`` -> the policy object. The module is imported from the working directory."""
    import importlib
    import sys
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("--policy takes module:attribute, e.g. my_adapter:policy")
    sys.path.insert(0, ".")
    obj = getattr(importlib.import_module(mod), attr)
    return obj() if isinstance(obj, type) else obj
