"""``python -m omnibase`` -- plan base placements for a recording, and say what it bought.

    python -m omnibase plan datasets/can_v3 --episode 6 --hands 0,1 --out plan.json
    python -m omnibase curve datasets/can_v3 --episode 6
    python -m omnibase robot so101
    python -m omnibase sweep datasets/can_v3 --workers 14 --out plans.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .data import (FRAMES, contacts, describe_dataset, episodes, fit_level, holds, load,
                   pick_hands)
from .plan import (SCORE_TERMS, arm_bases, ascii_map, base_grid, best_fixed, best_spot, home_index,
                   chunk, feasibility, home_share, mount_feasibility, mount_grid, pair,
                   score_map, solve, yield_curve)
from .robots import describe
from .robots import load as load_robot
from .write import frame_count, slice_video


def _grid(a):
    return base_grid(span=a.span, step=a.step, height=a.height)


def _home(a):
    """Where the robot really stands, if the caller said."""
    return None if not a.home else [float(v) for v in a.home.split(",")]


def _mount(a, hands):
    """A rigid assembly, if --pair asked for one. Returns ``(mount, cells)`` or ``(None, None)``.

    Two arms on one torso cannot be placed separately: the spacing is hardware, and the only
    choice is where the whole thing stands.
    """
    if not a.pair:
        return None, None
    if len(hands) != 2:
        raise SystemExit(f"--pair needs exactly two hands, this episode gave "
                         f"{[h.name for h in hands]}; narrow it with --hands")
    yaws = np.radians([float(v) for v in str(a.mount_yaw).split(",")])
    mount = pair(a.pair, names=[h.name for h in hands])
    return mount, mount_grid(span=a.span, step=a.step, height=a.height, yaws=yaws)


def _pick(ep, spec):
    try:
        return pick_hands(ep, spec)
    except ValueError as exc:
        raise SystemExit(str(exc))


def _frame_args(a):
    """The flags that say what the recording's numbers mean, as ``load`` keywords."""
    return dict(frame=getattr(a, "frame", None), level=getattr(a, "level", None),
                tcp=getattr(a, "tcp", None), stride=getattr(a, "stride", 1) or 1)


def _load(a):
    ep = load(a.dataset, episode=a.episode, chain=load_robot(a.robot), column=a.column,
              **_frame_args(a))
    return ep, _pick(ep, a.hands)


def cmd_plan(a):
    chain = load_robot(a.robot)
    ep, hands = _load(a)
    cells = _grid(a)
    print(f"episode {ep.index}: {ep.frames} frames, "
          f"{', '.join(repr(h.name) + (' (from joints)' if h.source == 'fk' else '') for h in hands)}"
          f", {len(cells)} candidate placements at height {a.height:+.3f} m")

    mount, mcells = _mount(a, hands)
    if mount is not None:
        cells = mcells
        combined, per = mount_feasibility(chain, [(h.pos, h.quat) for h in hands], mount, cells,
                                          a.pos_tol, np.radians(a.rot_tol))
        # One placement to choose, not one per arm: the assembly's.
        Fs, chunks, held = per, *chunk([combined], cells, window=a.window, home=_home(a))
        print(f"  rigid pair, {a.pair:.3f} m apart -- placing the assembly, not the arms")
    else:
        Fs = [feasibility(chain, h.pos, h.quat, cells, a.pos_tol, np.radians(a.rot_tol))
              for h in hands]
        chunks, held = chunk(Fs, cells, window=a.window, home=_home(a))

    print(f"\n{len(chunks)} chunk(s); "
          f"{'the assembly moves' if mount is not None else 'the bases move'} "
          f"{len(chunks) - 1} time(s):")
    for i, ch in enumerate(chunks):
        if mount is not None:
            c = ch.bases[0]
            at = "  ".join(f"{n} {np.round(b, 2).tolist()}"
                           for n, (b, _) in zip(mount.names, arm_bases(mount, c)))
            yaw = f" yaw {np.degrees(c[3]):+.0f}" if len(c) > 3 else ""
            bits = (f"mount ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}){yaw} -> {at}   "
                    f"both held {100 * ch.held[0]:5.1f}%")
        else:
            bits = "   ".join(f"{hands[k].name} ({b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f}) "
                              f"held {100 * h:5.1f}%"
                              for k, (b, h) in enumerate(zip(ch.bases, ch.held)))
        print(f"  chunk {i}: frames {ch.start:4d}-{ch.stop:4d} ({len(ch):4d})  {bits}")

    if a.home:
        share = home_share(chunks, ep.frames)
        print(f"\n  at the real base: "
              + ", ".join(f"{hands[k].name if mount is None else 'assembly'} "
                          f"{100 * s:.1f}% of frames"
                          for k, s in enumerate(share)))
        print("  the rest are retargeted somewhere your robot does not stand -- still training "
              "data, but\n  a claim about a different geometry. Chunk.at_home says which.")
    print()
    if mount is not None:
        cell, share = best_fixed(combined, cells)
        print(f"  BOTH arms at once: chunked {100 * held[0].mean():5.1f}% of frames held, "
              f"against {100 * share:5.1f}% for the best single assembly placement")
        for k, F in enumerate(Fs):
            print(f"    {hands[k].name:>6} alone would manage {100 * F.mean(0).max():5.1f}% "
                  "-- bolting them together costs the difference")
        return _write(a, ep, hands, chunks, mount)
    for k, F in enumerate(Fs):
        cell, share = best_fixed(F, cells)
        print(f"  {hands[k].name:>6}: chunked {100 * held[k].mean():5.1f}% of frames held, against "
              f"{100 * share:5.1f}% for the best single base "
              f"({cell[0]:+.2f}, {cell[1]:+.2f}, {cell[2]:+.2f})")

    return _write(a, ep, hands, chunks, None)


def _write(a, ep, hands, chunks, mount):
    if a.out:
        out = dict(dataset=str(a.dataset), episode=int(ep.index), robot=a.robot,
                   window=a.window, hands=[h.name for h in hands],
                   pair=(a.pair if mount is not None else None),
                   pos_tol=a.pos_tol, rot_tol_deg=a.rot_tol, height=a.height,
                   chunks=[dict(start=ch.start, stop=ch.stop,
                                bases=[[round(float(v), 4) for v in b] for b in ch.bases],
                                held=[round(float(h), 4) for h in ch.held]) for ch in chunks])
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


# ---- sweeping a whole dataset ---------------------------------------------------------------
# One chain per worker process, not one per episode: building the reach envelope costs a couple
# of seconds and is a property of the arm, so a worker that handles twenty episodes should pay
# for it once. Module-level because a spawned worker re-imports this module and fills it in.
_W = {}


def _worker(cfg):
    """Set a worker up once. The envelope is built on first use and then reused all run."""
    _W["cfg"] = cfg
    _W["chain"] = load_robot(cfg["robot"])
    _W["chain"].envelope()


def _sweep_one(episode):
    """Plan one episode. Returns a record, or an ``error`` record -- one bad episode in a
    hundred should not take the other ninety-nine down with it."""
    c, chain = _W["cfg"], _W["chain"]
    try:
        ep = load(c["dataset"], episode=episode, chain=chain, column=c["column"],
                  frame=c.get("frame"), level=c.get("level"), tcp=c.get("tcp"),
                  stride=c.get("stride", 1))
        hands = _pick(ep, c["hands"])
        cells = base_grid(span=c["span"], step=c["step"], height=c["height"])
        Fs = [feasibility(chain, h.pos, h.quat, cells, c["pos_tol"], np.radians(c["rot_tol"]))
              for h in hands]
        chunks, held = chunk(Fs, cells, window=c["window"], home=c["home"])
        return dict(episode=int(ep.index), frames=int(ep.frames),
                    hands=[h.name for h in hands],
                    solves=int(len(cells) * ep.frames * len(hands)),
                    # what `place` and `report` aggregate without loading the dataset again
                    reach=[F.sum(axis=0).astype(int).tolist() for F in Fs],
                    grasps=[int(len(contacts(h.grip, closing=True))) if h.grip is not None else None
                            for h in hands],
                    still=[int(holds(h.pos, h.grip).sum()) for h in hands],
                    extent=[[[round(float(v), 3) for v in h.pos.min(0)],
                             [round(float(v), 3) for v in h.pos.max(0)]] for h in hands],
                    held=[round(float(v), 4) for v in held.mean(axis=1)],
                    at_home=[round(float(v), 4) for v in home_share(chunks, ep.frames)],
                    fixed=[round(float(best_fixed(F, cells)[1]), 4) for F in Fs],
                    fixed_base=[[round(float(v), 4) for v in best_fixed(F, cells)[0]]
                                for F in Fs],
                    chunks=[dict(start=ch.start, stop=ch.stop,
                                 bases=[[round(float(v), 4) for v in b] for b in ch.bases],
                                 held=[round(float(h), 4) for h in ch.held],
                                 at_home=list(ch.at_home)) for ch in chunks])
    except Exception as exc:                       # noqa: BLE001 -- reported, not swallowed
        return dict(episode=int(episode), error=f"{type(exc).__name__}: {exc}")


def cmd_sweep(a):
    """Every episode in a dataset, in parallel, to one file.

    The parallelism is here rather than inside ``feasibility`` on purpose. Episodes are
    independent and there are many of them, so splitting here fills every core with no
    coordination, pays the pool's startup once for the whole run instead of once per call, and
    lets each worker keep its reach envelope. Splitting the cell sweep instead was measured at
    the same throughput per core and built a fresh pool for every episode.
    """
    import multiprocessing as mp
    import time

    # Flags this subcommand shares with the others but does not honour. Saying so beats
    # accepting them and quietly doing something else.
    if a.episode is not None:
        raise SystemExit("sweep plans every episode; name a subset with --episodes 0,1,2 "
                         "(plural), or plan just one with `omnibase plan --episode N`")
    if a.pair:
        raise SystemExit("sweep does not do rigid pairs yet; `omnibase plan --pair` does, "
                         "one episode at a time")
    eps = [int(v) for v in a.episodes.split(",")] if a.episodes else episodes(a.dataset)
    cfg = dict(dataset=str(a.dataset), robot=a.robot, column=a.column, hands=a.hands,
               span=a.span, step=a.step, height=a.height, window=a.window,
               pos_tol=a.pos_tol, rot_tol=a.rot_tol, home=_home(a),
               frame=a.frame, level=a.level, tcp=a.tcp, stride=a.stride)
    workers = max(1, min(a.workers, len(eps)))
    cells = len(base_grid(span=a.span, step=a.step, height=a.height))
    print(f"{len(eps)} episodes x {cells} placements, {workers} worker(s)", flush=True)

    t = time.time()
    if workers == 1:
        _worker(cfg)
        out = [_sweep_one(e) for e in _progress(eps, len(eps))]
    else:
        with mp.Pool(workers, initializer=_worker, initargs=(cfg,)) as pool:
            out = list(_progress(pool.imap_unordered(_sweep_one, eps), len(eps)))
    out.sort(key=lambda r: r["episode"])
    dt = time.time() - t

    good = [r for r in out if "error" not in r]
    bad = [r for r in out if "error" in r]
    solves = sum(r["solves"] for r in good)
    frames = sum(r["frames"] for r in good)
    print(f"\n{len(good)} episodes, {frames} frames, {solves} solves in {dt:.1f}s "
          f"({solves / max(dt, 1e-9) / 1000:.1f}k solves/s)")
    if good:
        # Weight by frames: a long episode is more of the dataset than a short one.
        def share(key):
            return sum(sum(r[key]) / max(len(r[key]), 1) * r["frames"]
                       for r in good) / max(frames, 1)
        print(f"  chunked {100 * share('held'):.1f}% of frames held, against "
              f"{100 * share('fixed'):.1f}% for the best single fixed base")
        if cfg["home"]:
            print(f"  {100 * share('at_home'):.1f}% of frames served from the robot's own base")
    for r in bad:
        print(f"  episode {r['episode']}: {r['error']}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(dataset=str(a.dataset), robot=a.robot, window=a.window, height=a.height,
                 pos_tol=a.pos_tol, rot_tol_deg=a.rot_tol, seconds=round(dt, 1),
                 cfg=cfg, episodes=out), indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


def _progress(it, total):
    """Say how far along we are. A dataset sweep is long enough to want it."""
    for i, item in enumerate(it, 1):
        print(f"\r  {i}/{total} episodes", end="", flush=True)
        yield item
    print()


def cmd_curve(a):
    chain = load_robot(a.robot)
    ep, hands = _load(a)
    cells = _grid(a)
    print(f"episode {ep.index}: how much survives, against how long one base must serve\n")
    print(f"  {'window':>12} {'usable':>8} {'placements':>12}")
    mount, mcells = _mount(a, hands)
    if mount is not None:
        cells = mcells
        combined, _ = mount_feasibility(chain, [(h.pos, h.quat) for h in hands], mount, cells,
                                        a.pos_tol, np.radians(a.rot_tol))
        series = [(f"both arms, rigid pair {a.pair:.2f} m", combined)]
    else:
        series = [(h.name, feasibility(chain, h.pos, h.quat, cells, a.pos_tol,
                                       np.radians(a.rot_tol))) for h in hands]
    for name, F in series:
        print(f"  {name}:")
        for row in yield_curve(F, cells):
            w = "whole episode" if row["window"] is None else f"{row['window']} frames"
            print(f"  {w:>12} {100 * row['usable']:7.1f}% "
                  f"{int(row['median_placements']):8d} median")


def cmd_map(a):
    chain = load_robot(a.robot)
    ep, hands = _load(a)
    cells = _grid(a)
    lo, hi = (0, None)
    if a.frames:
        lo, hi = (int(v) if v else None for v in a.frames.split(":"))
    for h in hands:
        p, q = h.pos[lo:hi], h.quat[lo:hi]
        if not a.frames:
            print("\nNote: scoring the WHOLE episode. One base rarely serves a whole "
                  "demonstration -- that is the problem this library exists for. Use "
                  "--frames a:b to score one chunk, or `omnibase plan` to cut the episode up.")
        smap = score_map(chain, p, q, cells, a.pos_tol, np.radians(a.rot_tol))
        cell, i, report = best_spot(smap)
        print(f"\n{h.name}, episode {ep.index}: stand at {report}")
        for name, why in SCORE_TERMS.items():
            print(f"    {name:<15} {smap[name][i]:.3f}   {why}")
        counts = {v: int((smap["verdict"] == v).sum()) for v in
                  ("good", "workable", "marginal", "unusable")}
        print("    verdicts: " + ", ".join(f"{v} {n}" for v, n in counts.items()))
        print(ascii_map(smap, a.key))


def cmd_robot(a):
    print(describe(load_robot(a.robot)))


def cmd_data(a):
    """What is in a dataset, and whether OmniBase can use it. Run this first."""
    print(describe_dataset(a.dataset, column=a.column, chain=load_robot(a.robot)))




def cmd_level(a):
    """Find which way is up in a recording. Run this before anything else on new data.

    A hand-held recording says how the rig moved, not where it was: FastUMI's poses are
    relative to the tracker's own pose at frame 0, so nothing in the file says where the floor
    is. What makes it recoverable is that frame 0 is a real place -- the gripper sits in a fixed
    slot on the collection rig before every take -- so the ORIENTATION is a property of the
    task. The frames where the jaw closed are the hand touching something that was standing on
    the table; the direction that makes those flat, within each episode, is up. The height is
    not shared and is read per episode at load time.
    """
    def report(euler, tcp, res, below, approach, score):
        print(f"    euler {euler:<4} tcp {tcp:.2f}   flat {1000 * res:5.1f} mm   "
              f"under {100 * below:5.1f}%   jaws {approach:5.1f} deg off the table   "
              f"score {score:+.4f}")

    print(f"{a.dataset}\n  fitting over up to {a.sample} episodes, "
          f"{len(a.eulers.split(','))}x{len(_floats(a.tcps))} conventions x tool offsets:")
    lvl = fit_level(a.dataset, frame=a.frame, column=a.column, sample=a.sample,
                    eulers=tuple(a.eulers.split(",")), tcps=tuple(_floats(a.tcps)),
                    tol=a.tol, report=report if a.verbose else None)
    print(f"\n  euler {lvl['euler']}, tool {lvl['tcp']:.2f} m ahead of what the rig tracked")
    print(f"  up {np.round(lvl['up'], 3).tolist()} in the recording's own axes")
    print(f"  {lvl['contacts']} contacts from {lvl['episodes']} episodes lie flat to "
          f"{lvl['flat']:.0f} mm rms (over the {100 * lvl['kept']:.0f}% kept)")
    print(f"  {100 * lvl['below']:.2f}% of all frames end up under the table")
    print(f"  fingers along {np.round(lvl['approach'], 3).tolist()} of the rig's own axes, and "
          f"they point at the table at {lvl['approach_deg']:.0f} deg when they close")
    print(f"  the same body direction does that at every grasp to {lvl['aligned']:.2f} of 1.00 -- "
          "which needs the episodes to share a frame, the angle order to be right")
    print("  and the fitted table to be the table, all at once")
    bad = []
    if lvl["flat"] > 30:
        bad.append("the contacts are not flat: these episodes do not share an orientation, so "
                   "there is no one 'up' for this task. Split it by episode range and fit each, "
                   "or drop the task -- do not average it")
    if lvl["aligned"] < 0.8:
        bad.append("no single body direction points at the table when the jaws close. Something "
                   "in the story is wrong -- the shared frame, the angle order, or the table")
    if lvl["below"] > 0.03:
        bad.append("too much of the recording ends up under the table: the convention is still "
                   "wrong, and every reachability answer built on it would be too")
    if lvl["approach_deg"] > 50:
        bad.append("the jaws never point at the table when they close, under any convention "
                   "tried. Either this task does not grasp downward -- watch a video -- or its "
                   "orientations use a convention that is not on the list")
    for b in bad:
        print(f"  WARNING: {b}")
    if not bad:
        print("  looks usable.")
    if a.out:
        Path(a.out).write_text(json.dumps(lvl, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


def _floats(spec):
    return [float(v) for v in str(spec).split(",")]


def _usable(rec, min_frames):
    """Frames a plan actually yields: chunks long enough to train on, weighted by what they hold."""
    return sum(len(rec["chunks"]) and (c["stop"] - c["start"]) * h
               for c in rec.get("chunks", []) if c["stop"] - c["start"] >= min_frames
               for h in c["held"])


def cmd_select(a):
    """Choose which episodes are worth downloading, given a byte budget.

    The point of sweeping metadata first: the poses are 30 kB an episode and the video is 10 MB,
    so every episode can be judged before any of them is fetched. What comes back is the subset
    with the most retargetable frames per byte -- which is not the same as the longest episodes,
    and not the same as the first N.
    """
    chosen, total_b, total_u = [], 0.0, 0.0
    pool = []
    for f in a.plans:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        root = Path(doc["dataset"])
        sizes = {}
        sz = root / "sizes.json"
        if sz.is_file():
            sizes = {int(k): int(v) for k, v in json.loads(sz.read_text()).items()}
        for rec in doc["episodes"]:
            if "error" in rec:
                continue
            u = _usable(rec, a.min_frames)
            if u <= 0 or (rec["frames"] and u / max(rec["frames"], 1) < a.min_held):
                continue
            b = float(sizes.get(rec["episode"], a.bytes_per_episode))
            pool.append(dict(dataset=str(root), episode=rec["episode"], bytes=b, usable=u,
                             frames=rec["frames"], plans=str(f)))
    pool.sort(key=lambda r: r["usable"] / max(r["bytes"], 1.0), reverse=True)
    budget = a.budget_gb * 1e9
    for r in pool:
        if total_b + r["bytes"] > budget:
            continue
        chosen.append(r)
        total_b += r["bytes"]
        total_u += r["usable"]
    per = {}
    for r in chosen:
        per[r["dataset"]] = per.get(r["dataset"], 0) + 1
    print(f"{len(pool)} episodes passed the filters, {len(chosen)} fit in {a.budget_gb:g} GB")
    for d, n in sorted(per.items()):
        print(f"  {n:6d}  {d}")
    print(f"  {total_b / 1e9:.1f} GB, {total_u / 1000:.0f}k usable frames "
          f"({total_u / max(total_b, 1) * 1e6:.1f} frames per MB)")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(budget_gb=a.budget_gb, total_bytes=total_b,
                                               total_usable=total_u, chosen=chosen), indent=1),
                               encoding="utf-8")
        print(f"\nwrote {a.out}")



# ---- turning a plan into a dataset -----------------------------------------------------------
# The plan says where a base would have to stand. That is an answer about the recording, and a
# policy cannot train on an answer -- it needs joint angles and the pictures that go with them.
# So the chosen placement is handed back to the solver, this time keeping what it returns, and
# each run of frames it holds becomes one episode of an ordinary LeRobot dataset.
#
# A chunk boundary is an episode boundary. The base moves between chunks, and a policy shown a
# single episode in which the robot teleports has been taught that robots teleport.


def _video_source(root, episode, hand, camera=None):
    """``(path, offset, fps)`` for one episode's clip -- both layouts, both naming schemes.

    v2.1 keeps one file per episode, so the offset is zero. v3.0 concatenates them and records
    where each begins; the offset is that, in frames, because the slice is cut by frame number.
    """
    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 0) or 30.0
    keys = [k for k, v in (info.get("features") or {}).items() if v.get("dtype") == "video"]
    if camera:
        keys = [k for k in keys if k == camera or k.endswith(camera)]
    elif len(keys) > 1 and hand:
        keys = [k for k in keys if hand.lower() in k.lower()] or keys
    if not keys:
        return None, 0, fps
    key = keys[0]
    tpl = info.get("video_path") or ""
    if "episode_index" in tpl:                       # v2.1: one file per episode
        rel = tpl.format(episode_chunk=episode // int(info.get("chunks_size", 1000)),
                         video_key=key, episode_index=episode)
        return root / rel, 0, fps
    rows = pd.read_parquet(next((root / "meta" / "episodes").glob("*/*.parquet")))
    row = rows[rows["episode_index"] == episode]
    if row.empty:
        return None, 0, fps
    row = row.iloc[0]
    rel = tpl.format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                     file_index=int(row[f"videos/{key}/file_index"]))
    return root / rel, int(round(float(row[f"videos/{key}/from_timestamp"]) * fps)), fps


def _runs(ok, least):
    """Maximal stretches of True at least ``least`` long: ``[(start, stop), ...]``."""
    out, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= least:
            out.append((i, j))
        i = j
    return out


_E = {}


def _export_worker(cfg):
    _E["cfg"] = cfg
    _E["chain"] = load_robot(cfg["sources"][0]["cfg"].get("robot", "so101"))
    _E["chain"].envelope()


def _export_one(job):
    """One source episode -> zero or more output episodes, video already cut."""
    si, rec = job
    c = _E["cfg"]
    src = c["sources"][si]
    cfg, chain = src["cfg"], _E["chain"]
    stride = int(cfg.get("stride", 1) or 1)
    out = []
    try:
        ep = load(cfg["dataset"], episode=rec["episode"], chain=chain, column=cfg["column"],
                  frame=cfg.get("frame"), level=cfg.get("level"), tcp=cfg.get("tcp"),
                  stride=stride)
        hands = _pick(ep, cfg.get("hands"))
        chunks = rec["chunks"]
        if c["fixed_base"]:
            chunks = [dict(start=0, stop=rec["frames"], bases=rec["fixed_base"])]
        lo, hi = src["grip"]
        for a, hand in enumerate(hands):
            vid, offset, _ = _video_source(cfg["dataset"], rec["episode"], hand.name,
                                           src.get("camera"))
            if vid is None or not Path(vid).exists():
                continue
            for ci, ch in enumerate(chunks):
                s0, s1 = int(ch["start"]), int(ch["stop"])
                base = np.asarray(ch["bases"][a], dtype=float)
                q, ok = solve(chain, hand.pos[s0:s1], hand.quat[s0:s1], base,
                              cfg["pos_tol"], np.radians(cfg["rot_tol"]))
                # A step no arm could take in one frame ends the episode there rather than
                # being written down. These are real -- the SO-101's wrist stops at -157 and
                # +163 degrees, so a hand that rotates past the gap leaves the arm to unwind
                # 320 the other way -- and both frames either side are honest solutions. What
                # is not honest is the line between them, and a policy would learn the flick.
                if c["max_step"] and len(q) > 1:
                    ok[1:] &= np.abs(np.diff(np.degrees(q), axis=0)).max(1) <= c["max_step"]
                for ri, (r0, r1) in enumerate(_runs(ok, c["min_frames"])):
                    qa = np.degrees(q[r0:r1])
                    g = hand.grip[s0 + r0:s0 + r1] if hand.grip is not None else np.zeros(r1 - r0)
                    g = np.clip((np.asarray(g, dtype=float) - lo) / (hi - lo), 0.0, 1.0) * 100.0
                    state = np.concatenate([qa, g[:, None]], axis=1).astype(np.float32)
                    # The action is where the arm goes NEXT. The last frame has nowhere to go,
                    # so it holds -- which is also what the arm does at the end of an episode.
                    action = np.vstack([state[1:], state[-1:]])
                    if c.get("action_space") == "delta":
                        # next minus current for the joints; the jaw stays a target. Measured
                        # (docs/experiment, section 11): a base shift of 8 cm costs absolute
                        # targets 8 cm at the tool and deltas under 1 cm.
                        action = action.copy()
                        action[:, :-1] -= state[:, :-1]
                    stage = Path(c["stage"]) / f"{si}_{rec['episode']}_{a}_{ci}_{ri}.mp4"
                    slice_video(vid, stage, (s0 + r0) * stride, (s0 + r1) * stride, stride,
                                c["fps"], c["size"], crf=c["crf"], offset=offset)
                    n = frame_count(stage)
                    if n != len(state):
                        out.append(dict(error=f"episode {rec['episode']} hand {hand.name} "
                                              f"chunk {ci}: {n} video frames for "
                                              f"{len(state)} rows"))
                        Path(stage).unlink(missing_ok=True)
                        continue
                    out.append(dict(src=si, episode=int(rec["episode"]), hand=hand.name,
                                    chunk=ci, run=ri, start=s0 + r0,
                                    state=state, action=action, video=str(stage),
                                    task=src["task"] or ep_task(cfg["dataset"]),
                                    base=[round(float(v), 4) for v in base],
                                    jump=float(np.abs(np.diff(qa, axis=0)).max()
                                               if len(qa) > 1 else 0.0)))
    except Exception as exc:                          # noqa: BLE001 -- one episode, not the run
        out.append(dict(error=f"episode {rec.get('episode')}: {type(exc).__name__}: {exc}"))
    return out


def ep_task(root):
    """The language string a source dataset already carries, so it need not be retyped."""
    root = Path(root)
    j = root / "meta" / "tasks.jsonl"
    if j.is_file():
        for line in j.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line)["task"]
    q = root / "meta" / "tasks.parquet"
    if q.is_file():
        import pandas as pd
        return str(pd.read_parquet(q).index[0])
    return "manipulate the object"


def _per_source(values, n, name, cast=str):
    """A flag given once applies to every source; given n times, one each."""
    if not values:
        return [None] * n
    if len(values) == 1:
        return [cast(values[0])] * n
    if len(values) != n:
        raise SystemExit(f"--{name} given {len(values)} times for {n} plans file(s): "
                         "pass it once for all of them, or once per file, in order")
    return [cast(v) for v in values]


def cmd_export(a):
    """Retarget planned episodes into a LeRobot dataset an arm could be trained on."""
    import multiprocessing as mp
    import time

    from .write import Writer

    docs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in a.plans]
    for f, d in zip(a.plans, docs):
        if "cfg" not in d:
            raise SystemExit(f"{f} was written by an older sweep and does not say how the data "
                             "was read; re-run `omnibase sweep --out` to produce it")
    n = len(docs)
    tasks = _per_source(a.task, n, "task")
    cameras = _per_source(a.camera, n, "camera")
    repeats = _per_source(a.repeat, n, "repeat", int)
    grips = _per_source(a.grip, n, "grip", lambda v: tuple(float(x) for x in str(v).split(",")))
    for i, (d, g) in enumerate(zip(docs, grips)):
        if g is None:
            preset = FRAMES.get(d["cfg"].get("frame") or "", {})
            lvl = d["cfg"].get("level")
            if lvl and Path(lvl).is_file():
                preset = {**preset, **{k: v for k, v in
                                       json.loads(Path(lvl).read_text()).items() if v}}
            grips[i] = tuple(preset.get("grip") or ())
            if len(grips[i]) != 2:
                raise SystemExit(f"{a.plans[i]}: no gripper range known for this source. Pass "
                                 "--grip lo,hi -- the closed and open values its column uses.")
    sources = [dict(cfg=d["cfg"], task=t, camera=c, grip=list(g))
               for d, t, c, g in zip(docs, tasks, cameras, grips)]

    stage = Path(a.out).parent / f".{Path(a.out).name}_stage"
    stage.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in a.size.lower().split("x"))
    cfg = dict(sources=sources, stage=str(stage), fps=a.fps, size=(w, h), crf=a.crf,
               min_frames=a.min_frames, fixed_base=bool(a.fixed_base), max_step=a.max_step,
               action_space=a.action)

    jobs = [(i, rec) for i, d in enumerate(docs) for rec in d["episodes"]
            if "error" not in rec and (not a.fixed_base or rec.get("fixed_base"))]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} planned episode(s) from {n} source(s) -> {a.out}"
          f"{'  [single fixed base]' if a.fixed_base else ''}", flush=True)

    t0 = time.time()
    workers = max(1, min(a.workers, len(jobs)))
    if workers == 1:
        _export_worker(cfg)
        got = [r for job in _progress(jobs, len(jobs)) for r in _export_one(job)]
    else:
        with mp.Pool(workers, initializer=_export_worker, initargs=(cfg,)) as pool:
            got = [r for out in _progress(pool.imap_unordered(_export_one, jobs), len(jobs))
                   for r in out]

    bad = [r for r in got if "error" in r]
    good = sorted((r for r in got if "error" not in r),
                  key=lambda r: (r["src"], r["episode"], r["hand"], r["start"]))
    if not good:
        raise SystemExit("nothing was exported" + (f"; first problem: {bad[0]['error']}" if bad else ""))

    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos",
             "wrist_roll.pos", "gripper.pos"]
    out = Writer(a.out, fps=a.fps, names=names, video_key=f"observation.images.{a.key}",
                 size=(w, h), robot_type=sources[0]["cfg"].get("robot", "so101"),
                 meta=dict(action_space=a.action, action_note=(
                     "joint deltas: action[:5] = next joints - current joints (degrees); "
                     "gripper.pos is a target" if a.action == "delta" else
                     "absolute joint targets in degrees; gripper.pos 0 closed .. 100 open")))
    for r in good:
        i = out.add(r["state"], r["action"], r["task"], r["video"])
        rep = repeats[r["src"]] or 1
        if rep > 1:
            out.duplicate(i, rep - 1)
    root = out.finish()
    shutil.rmtree(stage, ignore_errors=True)

    jumps = np.array([r["jump"] for r in good])
    print(f"\n{out.frames} frames in {len(out.episodes)} episodes "
          f"({len(good)} before repeats), {len(out.tasks)} task(s), {time.time() - t0:.0f}s")
    print(f"  per-frame joint step: median {np.median(jumps):.1f} deg, "
          f"p95 {np.percentile(jumps, 95):.1f}, worst {jumps.max():.1f}")
    if jumps.max() > 45:
        print("  (a large worst-case step is the solver changing posture between frames; "
              "it shows up as a flick in one episode, not a wrong base)")
    for e in bad[:10]:
        print(f"  dropped: {e['error']}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")
    print(f"\nwrote {root}")



# ---- what ships as one command each --------------------------------------------------------

def _doc(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if "cfg" not in doc:
        raise SystemExit(f"{path}: written by an older sweep; re-run `omnibase sweep --out`")
    return doc


def _sweep_doc(a):
    """A plan document: the file given, or a fresh sweep of the dataset written to --out."""
    if str(a.dataset).lower().endswith(".json"):
        return _doc(a.dataset)
    if not a.out:
        a.out = str(Path(a.dataset).with_suffix("")) + "_plans.json"
    cmd_sweep(a)
    return _doc(a.out)


def placement(doc):
    """Executable frames per base, summed over a sweep: ``(cells, per_hand, frames, totals)``.

    ``per_hand[name]`` is a ``(cells,)`` array of frames that hand can hold from each base;
    ``totals`` has the dataset-wide frame count and the chunked / best-fixed / at-home yields.
    """
    cfg = doc["cfg"]
    cells = base_grid(span=cfg["span"], step=cfg["step"], height=cfg["height"])
    good = [r for r in doc["episodes"] if "error" not in r and r.get("reach")]
    if not good:
        raise SystemExit("this sweep has no per-cell counts; re-run `omnibase sweep --out` with this version")
    per, frames = {}, {}
    for r in good:
        for name, reach in zip(r["hands"], r["reach"]):
            per[name] = per.get(name, np.zeros(len(cells))) + np.asarray(reach, float)
            frames[name] = frames.get(name, 0) + r["frames"]
    tot = sum(r["frames"] for r in good)

    def share(key):
        return sum(sum(r[key]) / max(len(r[key]), 1) * r["frames"] for r in good) / max(tot, 1)

    totals = dict(episodes=len(good), frames=tot, chunked=share("held"), fixed_per_episode=share("fixed"),
                  at_home=share("at_home") if cfg.get("home") else None)
    return cells, per, frames, totals


def cmd_place(a):
    """Where should the robot stand for this dataset? Executable frames from every base."""
    doc = _sweep_doc(a)
    cells, per, frames, totals = placement(doc)
    cfg = doc["cfg"]
    result = dict(dataset=doc["dataset"], robot=doc["robot"], cells=cells.tolist(),
                  pos_tol=cfg["pos_tol"], rot_tol_deg=cfg["rot_tol"], totals=totals, hands={})
    print(f"{totals['episodes']} episodes, {totals['frames']} frames, {len(cells)} candidate bases "
          f"(step {cfg['step']:g} m, height {cfg['height']:g} m)\n")
    for name, counts in per.items():
        frac = counts / max(frames[name], 1)
        order = np.argsort(frac)[::-1]
        best = int(order[0])
        print(f"hand {name!r}: {frames[name]} frames")
        print(f"  stand at ({cells[best][0]:+.2f}, {cells[best][1]:+.2f}, {cells[best][2]:+.2f}) m: "
              f"{100 * frac[best]:.1f}% of frames executable from one fixed base")
        for i in order[1:a.top]:
            print(f"         ({cells[i][0]:+.2f}, {cells[i][1]:+.2f}, {cells[i][2]:+.2f}) m: {100 * frac[i]:.1f}%")
        home = None
        if cfg.get("home"):
            h = home_index(cells, cfg["home"])
            home = dict(base=cfg["home"], fraction=float(frac[h]))
            print(f"  from the robot's own base {tuple(cfg['home'])}: {100 * frac[h]:.1f}%")
        print(ascii_map(dict(cells=cells, coverage=frac, score=frac), "score"))
        result["hands"][name] = dict(frames=frames[name], executable=counts.astype(int).tolist(),
                                     best=dict(base=cells[best].tolist(), fraction=float(frac[best])), home=home)
        print()
    print(f"whole dataset: best fixed base per episode holds {100 * totals['fixed_per_episode']:.1f}% of frames, "
          f"one base per {cfg['window']}-frame window holds {100 * totals['chunked']:.1f}%")
    if a.place_out:
        Path(a.place_out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {a.place_out}")


def report(doc):
    """A dataset, as seen by a robot: what a fixed base loses, what the demonstrations contain."""
    cfg = doc["cfg"]
    cells, per, frames, totals = placement(doc)
    good = [r for r in doc["episodes"] if "error" not in r]
    bad = [r for r in doc["episodes"] if "error" in r]
    L = [f"# {Path(doc['dataset']).name} as seen by {doc['robot']}", "",
         f"{len(good)} episodes ({len(bad)} unreadable), {totals['frames']} frames, "
         f"tolerances {1000 * cfg['pos_tol']:g} mm / {cfg['rot_tol']:g} deg, window {cfg['window']} frames.", "",
         "## Yield", "", "| | frames held |", "|---|---|"]
    for name, counts in per.items():
        frac = counts / max(frames[name], 1); b = int(frac.argmax())
        L.append(f"| best single fixed base for the whole dataset, hand {name!r} at ({cells[b][0]:+.2f}, {cells[b][1]:+.2f}) | {100 * frac[b]:.1f}% |")
    L.append(f"| best fixed base chosen per episode | {100 * totals['fixed_per_episode']:.1f}% |")
    L.append(f"| one base per {cfg['window']}-frame window (OmniBase) | {100 * totals['chunked']:.1f}% |")
    if cfg.get("home"):
        L.append(f"| the robot's own base {tuple(cfg['home'])}, windows served from there | {100 * totals['at_home']:.1f}% |")
    chunks = [c for r in good for c in r.get("chunks", [])]
    lens = [c["stop"] - c["start"] for c in chunks]
    L += ["", "## What the demonstrations contain", "", "| | |", "|---|---|",
          f"| chunks (one base each) | {len(chunks)}, median {int(np.median(lens)) if lens else 0} frames |"]
    for i, name in enumerate(per):
        gr = [r["grasps"][i] for r in good if r.get("grasps") and r["grasps"][i] is not None]
        st = [r["still"][i] for r in good if r.get("still")]
        ex = np.array([r["extent"][i] for r in good if r.get("extent")])
        if gr:
            L.append(f"| grasp events, hand {name!r} | {sum(gr)} ({np.mean(gr):.1f} per episode) |")
        if st:
            L.append(f"| still frames, hand {name!r} (moved < 2 mm, jaw unchanged) | {sum(st)} = {100 * sum(st) / max(frames[name], 1):.0f}% of frames |")
        if len(ex):
            lo, hi = ex[:, 0].min(0), ex[:, 1].max(0)
            L.append(f"| workspace, hand {name!r} | x {lo[0]:+.2f}..{hi[0]:+.2f}, y {lo[1]:+.2f}..{hi[1]:+.2f}, z {lo[2]:+.2f}..{hi[2]:+.2f} m |")
    L += ["", "## Executable frames per base", ""]
    for name, counts in per.items():
        frac = counts / max(frames[name], 1)
        L += [f"hand {name!r}:", "", "```", ascii_map(dict(cells=cells, coverage=frac, score=frac), "score"), "```", ""]
    if bad:
        L += ["## Unreadable episodes", ""] + [f"- episode {r['episode']}: {r['error']}" for r in bad[:20]]
    return "\n".join(L)


def cmd_report(a):
    md = report(_doc(a.plans))
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n", encoding="utf-8"); print(f"\nwrote {a.out}")


def cmd_ambiguity(a):
    from .eval import ambiguity, ambiguity_table, chunks_from_plan
    chunks, chain, cfg = chunks_from_plan(a.plans, hands=a.hands)
    if a.limit:
        chunks = chunks[: a.limit]
    print(f"{len(chunks)} chunks, {sum(len(c['pos']) for c in chunks)} frames; "
          f"what averaging the joint actions of a base grid costs at the tool:\n")
    rows = ambiguity(chain, chunks, _floats(a.spans), a.zspan, cfg["pos_tol"], np.radians(cfg["rot_tol"]))
    print(ambiguity_table(rows))
    print("\nRule: augment bases only with joint deltas or end-effector actions, unless the policy "
          "can tell the bases apart from its state.")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8"); print(f"\nwrote {a.out}")


def cmd_probe(a):
    from .eval import chunks_from_plan, load_policy, probe, probe_table
    chunks, chain, cfg = chunks_from_plan(a.plans, hands=a.hands)
    if a.limit:
        chunks = chunks[: a.limit]
    policies = {p: load_policy(p) for p in a.policy}
    res = probe(chain, chunks, policies, _floats(a.offsets), cfg["pos_tol"], np.radians(cfg["rot_tol"]),
                grasp_rows=a.grasp_rows, knn_grid_cm=_floats(a.knn_grid) if a.knn_grid else None)
    print(f"{len(chunks)} chunks, {res['rows']['0']['frames']} frame pairs at the plan's bases. "
          f"Median miss of the predicted next tool point when the base moves:\n")
    print(probe_table(res, "miss_cm"))
    if res["rows"]["0"].get("grasp_frames"):
        print(f"\nWithin {a.grasp_rows} rows of the jaws closing ({res['rows']['0']['grasp_frames']} frames):\n")
        print(probe_table(res, "grasp_cm"))
    if any("nll" in v for v in res["rows"]["0"].values() if isinstance(v, dict)):
        print("\nLikelihood of the re-solved action (nats; lower is more likely):\n"); print(probe_table(res, "nll"))
        print("\nLikelihood of the UNSHIFTED action at the shifted state -- should fall as the shift grows:\n")
        print(probe_table(res, "nll_replay"))
    print("\nReplay misses by exactly the shift; the oracle by nothing. A policy near the replay line "
          "memorised its base. This does not predict success (see docs).")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8"); print(f"\nwrote {a.out}")


def cmd_serve(a):
    try:
        import uvicorn
        from .service import app
    except ImportError:
        raise SystemExit("the service needs fastapi and uvicorn: pip install 'omnibase[service]'")
    uvicorn.run(app, host=a.host, port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser("omnibase", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, data=True):
        if data:
            q.add_argument("dataset")
            q.add_argument("--episode", type=int, default=None)
            q.add_argument("--hands", default=None,
                           help="which hands, by name (right,left) or index. Default: all of "
                                "them. `omnibase data <path>` lists what a dataset has.")
            q.add_argument("--column", default="action",
                           choices=("action", "observation.state"),
                           help="commanded poses or measured ones")
            q.add_argument("--home", default=None, metavar="X,Y,Z",
                           help="where the robot ACTUALLY stands. Given it, the real base is "
                                "used wherever it works and left only where it cannot, coming "
                                "back at the first frame it can serve again. Frames retargeted "
                                "to it need no explanation at deployment.")
            q.add_argument("--pair", type=float, default=None, metavar="METRES",
                           help="the two arms are rigidly coupled this far apart -- a torso, a "
                                "humanoid, one base plate. Places the assembly instead of the "
                                "arms, so a placement only counts where BOTH can reach.")
            q.add_argument("--mount-yaw", default="0", metavar="DEG,DEG",
                           help="mount headings to try with --pair. Turning a torso swings the "
                                "far arm through an arc, so this is a real degree of freedom "
                                "in a way a single arm's yaw is not.")
        if data:
            q.add_argument("--frame", default=None, choices=sorted(FRAMES),
                           help="the rig this was recorded with, when it is not this library's "
                                "frame: its body axes, Euler convention and tool offset.")
            q.add_argument("--level", default=None, metavar="LEVEL.json",
                           help="a scene calibration from `omnibase level` -- where the table "
                                "is, and the convention that revealed it. Carries the rest of "
                                "the recipe, so it is the only frame flag you need once fitted.")
            q.add_argument("--tcp", type=float, default=None, metavar="M",
                           help="metres from whatever the rig tracked to the point between the "
                                "fingers. Overrides the preset's.")
            q.add_argument("--stride", type=int, default=1, metavar="N",
                           help="keep every n-th frame: retarget at the rate the policy will "
                                "run at, so --window means the same thing in both.")
        q.add_argument("--robot", default="so101")
        q.add_argument("--window", type=int, default=21,
                       help="frames one base must cover: your policy's observation buffer plus "
                            "its action chunk. The whole lever.")
        q.add_argument("--pos-tol", type=float, default=0.015, metavar="M")
        q.add_argument("--rot-tol", type=float, default=20.0, metavar="DEG")
        q.add_argument("--span", type=float, default=0.42, metavar="M")
        q.add_argument("--step", type=float, default=0.02, metavar="M")
        q.add_argument("--height", type=float, default=0.08, metavar="M")

    q = sub.add_parser("plan", help="chunk a recording into base placements")
    common(q)
    q.add_argument("--out", default=None, metavar="PLAN.json")
    q.set_defaults(func=cmd_plan)

    q = sub.add_parser("sweep", help="plan every episode in a dataset, in parallel")
    common(q)
    q.add_argument("--workers", type=int, default=1, metavar="N",
                   help="processes to split the episodes across. Episodes are independent, so "
                        "this scales with cores until you run out of episodes.")
    q.add_argument("--episodes", default=None, metavar="I,J,K",
                   help="only these episodes. Default: every episode the dataset declares.")
    q.add_argument("--out", default=None, metavar="PLANS.json")
    q.set_defaults(func=cmd_sweep)

    q = sub.add_parser("place", help="where should the robot stand? executable frames from every base, over a dataset")
    common(q)
    q.add_argument("--workers", type=int, default=1, metavar="N")
    q.add_argument("--episodes", default=None, metavar="I,J,K")
    q.add_argument("--out", default=None, metavar="PLANS.json",
                   help="where the sweep is written (default: next to the dataset). Give a PLANS.json "
                        "as the dataset to reuse one.")
    q.add_argument("--place-out", default=None, metavar="PLACE.json", help="the map and the best bases, as JSON")
    q.add_argument("--top", type=int, default=5, help="how many runner-up bases to list")
    q.set_defaults(func=cmd_place)

    q = sub.add_parser("report", help="a dataset as a robot sees it: yield, grasps, pauses, workspace, the base map")
    q.add_argument("plans", metavar="PLANS.json", help="from `omnibase sweep --out` or `omnibase place`")
    q.add_argument("--out", default=None, metavar="REPORT.md")
    q.set_defaults(func=cmd_report)

    q = sub.add_parser("ambiguity", help="is base augmentation safe for this action space? miss in cm, no training")
    q.add_argument("plans", metavar="PLANS.json")
    q.add_argument("--spans", default="2,4,6", metavar="CM,CM", help="half-widths of the 3x3 base grids to test")
    q.add_argument("--zspan", type=float, default=0.0, metavar="CM", help="also +- this in height (3x3x3)")
    q.add_argument("--hands", default=None)
    q.add_argument("--limit", type=int, default=None, help="first N chunks only")
    q.add_argument("--out", default=None, metavar="OUT.json")
    q.set_defaults(func=cmd_ambiguity)

    q = sub.add_parser("probe", help="does a policy know where it stands? miss in cm against a base shift, no rollout")
    q.add_argument("plans", metavar="PLANS.json")
    q.add_argument("--policy", action="append", default=[], metavar="module:attr",
                   help="a callable policy(state, where) -> action in radians; may carry .nll(state, action, where). Repeatable.")
    q.add_argument("--knn-grid", default="-6,-3,0,3,6", metavar="CM,..",
                   help="add two nearest-neighbour references, fit at one base and on this grid; '' for none")
    q.add_argument("--offsets", default="2,4,6,8", metavar="CM,CM")
    q.add_argument("--grasp-rows", type=int, default=5)
    q.add_argument("--hands", default=None)
    q.add_argument("--limit", type=int, default=None, help="first N chunks only")
    q.add_argument("--out", default=None, metavar="OUT.json")
    q.set_defaults(func=cmd_probe)

    q = sub.add_parser("serve", help="run the web service and UI (pip install 'omnibase[service]')")
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=8000)
    q.set_defaults(func=cmd_serve)

    q = sub.add_parser("curve", help="yield against how long one base must serve")
    common(q)
    q.set_defaults(func=cmd_curve)

    q = sub.add_parser("map", help="score every base placement and draw the map")
    common(q)
    q.add_argument("--frames", default=None, metavar="A:B",
                   help="score only these frames -- a chunk, rather than the whole episode")
    q.add_argument("--key", default="score",
                   choices=("score", "coverage", "limit_margin", "manipulability", "rot_margin"),
                   help="which component to draw")
    q.set_defaults(func=cmd_map)

    q = sub.add_parser("level", help="find the table a recording was made on")
    common(q)
    q.add_argument("--sample", type=int, default=150, metavar="N",
                   help="episodes to pool. One episode has a handful of contact points; a "
                        "plane needs more than that to be believed.")
    q.add_argument("--eulers", default="xyz",
                   help="conventions to try. Lowercase is extrinsic (roll-pitch-yaw as most "
                        "rigs mean it), uppercase intrinsic.")
    q.add_argument("--tcps", default="0,0.03,0.06,0.09,0.12,0.15",
                   help="tool offsets to try, metres. Only the right one collapses grasps made "
                        "at different wrist angles onto one plane.")
    q.add_argument("--tol", type=float, default=0.04, metavar="M",
                   help="how far a contact point may sit off the plane and still count -- half "
                        "an object, roughly.")
    q.add_argument("--verbose", action="store_true", help="print every candidate")
    q.add_argument("--out", default=None, metavar="LEVEL.json")
    q.set_defaults(func=cmd_level)

    q = sub.add_parser("select", help="choose episodes worth downloading, within a byte budget")
    common(q, data=False)
    q.add_argument("plans", nargs="+", metavar="PLANS.json")
    q.add_argument("--budget-gb", type=float, default=90.0)
    q.add_argument("--bytes-per-episode", type=float, default=10e6,
                   help="fallback when the dataset has no sizes.json")
    q.add_argument("--min-held", type=float, default=0.9,
                   help="skip an episode that yields less than this share of its frames")
    q.add_argument("--min-frames", type=int, default=21,
                   help="ignore chunks shorter than this: below a policy's action chunk they "
                        "are all padding")
    q.add_argument("--out", default=None, metavar="SELECTION.json")
    q.set_defaults(func=cmd_select)

    q = sub.add_parser("export", help="retarget planned episodes into a LeRobot dataset")
    common(q, data=False)
    q.add_argument("plans", nargs="+", metavar="PLANS.json")
    q.add_argument("--out", required=True, metavar="DIR")
    q.add_argument("--fps", type=int, required=True,
                   help="the rate of the exported rows. With --stride on the sweep, this is the "
                        "source rate divided by it.")
    q.add_argument("--task", action="append", default=[], metavar="STRING",
                   help="language string for a source. Default: what the source dataset says. "
                        "Give once for all, or once per plans file.")
    q.add_argument("--camera", action="append", default=[], metavar="KEY",
                   help="which camera to take, when a source has several")
    q.add_argument("--grip", action="append", default=[], metavar="LO,HI",
                   help="the closed and open values of the source's gripper column")
    q.add_argument("--repeat", action="append", default=[], metavar="N",
                   help="write a source's episodes N times. Sixteen episodes of the task you "
                        "care about, among ten thousand of something else, are 0.2%% of the "
                        "sampler's attention.")
    q.add_argument("--action", default="absolute", choices=("absolute", "delta"),
                   help="absolute joint targets, or next-minus-current joint deltas. Deltas "
                        "survive a base shift; absolute targets carry the base inside them.")
    q.add_argument("--fixed-base", action="store_true",
                   help="ignore the chunking and stand the arm in one place for the whole "
                        "episode, keeping only the frames it can hold. The honest comparison.")
    q.add_argument("--size", default="512x384", metavar="WxH")
    q.add_argument("--key", default="wrist", metavar="NAME",
                   help="camera name in the OUTPUT dataset (observation.images.NAME)")
    q.add_argument("--min-frames", type=int, default=20,
                   help="drop an exported run shorter than this: below the policy's action "
                        "chunk it is all padding")
    q.add_argument("--max-step", type=float, default=60.0, metavar="DEG",
                   help="end an episode where any joint would have to move more than this in "
                        "one frame. Mostly the wrist unwinding through its own dead zone, "
                        "which is a real motion and not one to train on. 0 keeps everything.")
    q.add_argument("--crf", type=int, default=23, help="x264 quality, lower is bigger")
    q.add_argument("--limit", type=int, default=None, help="only the first N episodes, to try it")
    q.add_argument("--workers", type=int, default=1, metavar="N")
    q.set_defaults(func=cmd_export)

    q = sub.add_parser("data", help="what a dataset holds, and whether it can be used")
    common(q)
    q.set_defaults(func=cmd_data)

    q = sub.add_parser("robot", help="print a built-in arm, to check it reads right")
    common(q, data=False)
    q.set_defaults(func=cmd_robot)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
