"""``python -m omnibase`` -- plan base placements for a recording, and say what it bought.

    python -m omnibase plan datasets/can_v3 --episode 6 --hands 0,1 --out plan.json
    python -m omnibase curve datasets/can_v3 --episode 6
    python -m omnibase robot so101
    python -m omnibase sweep datasets/can_v3 --workers 14 --out plans.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import describe_dataset, episodes, load
from .plan import (SCORE_TERMS, arm_bases, ascii_map, base_grid, best_fixed, best_spot,
                   chunk, feasibility, home_share, mount_feasibility, mount_grid, pair,
                   score_map, yield_curve)
from .robots import describe
from .robots import load as load_robot


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
    """The episode's hands, filtered by a --hands string, which takes names or indices."""
    hands = list(ep.hands)
    if not spec:
        return hands
    chosen = []
    for w in [w.strip() for w in spec.split(",") if w.strip()]:
        if w.isdigit() and int(w) < len(hands):
            chosen.append(hands[int(w)])
        else:
            match = [h for h in hands if h.name == w]
            if not match:
                raise SystemExit(f"no hand {w!r}; this episode has {[h.name for h in hands]}")
            chosen.append(match[0])
    return chosen


def _load(a):
    ep = load(a.dataset, episode=a.episode, chain=load_robot(a.robot), column=a.column)
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
        ep = load(c["dataset"], episode=episode, chain=chain, column=c["column"])
        hands = _pick(ep, c["hands"])
        cells = base_grid(span=c["span"], step=c["step"], height=c["height"])
        Fs = [feasibility(chain, h.pos, h.quat, cells, c["pos_tol"], np.radians(c["rot_tol"]))
              for h in hands]
        chunks, held = chunk(Fs, cells, window=c["window"], home=c["home"])
        return dict(episode=int(ep.index), frames=int(ep.frames),
                    hands=[h.name for h in hands],
                    solves=int(len(cells) * ep.frames * len(hands)),
                    held=[round(float(v), 4) for v in held.mean(axis=1)],
                    at_home=[round(float(v), 4) for v in home_share(chunks, ep.frames)],
                    fixed=[round(float(best_fixed(F, cells)[1]), 4) for F in Fs],
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
               pos_tol=a.pos_tol, rot_tol=a.rot_tol, home=_home(a))
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
                 episodes=out), indent=1), encoding="utf-8")
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
