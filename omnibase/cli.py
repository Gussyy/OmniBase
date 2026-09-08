"""``python -m omnibase`` -- plan base placements for a recording, and say what it bought.

    python -m omnibase plan datasets/can_v3 --episode 6 --hands 0,1 --out plan.json
    python -m omnibase curve datasets/can_v3 --episode 6
    python -m omnibase robot so101
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import describe_dataset, load
from .plan import (SCORE_TERMS, ascii_map, base_grid, best_fixed, best_spot,
                   chunk, feasibility, score_map, yield_curve)
from .robots import describe
from .robots import load as load_robot


def _grid(a):
    return base_grid(span=a.span, step=a.step, height=a.height)


def _load(a):
    """The episode's hands, filtered by --hands, which takes names or indices."""
    ep = load(a.dataset, episode=a.episode, chain=load_robot(a.robot), column=a.column)
    hands = list(ep.hands)
    if a.hands:
        want = [w.strip() for w in a.hands.split(",") if w.strip()]
        chosen = []
        for w in want:
            if w.isdigit() and int(w) < len(hands):
                chosen.append(hands[int(w)])
            else:
                match = [h for h in hands if h.name == w]
                if not match:
                    raise SystemExit(f"no hand {w!r}; this episode has "
                                     f"{[h.name for h in hands]}")
                chosen.append(match[0])
        hands = chosen
    return ep, hands


def cmd_plan(a):
    chain = load_robot(a.robot)
    ep, hands = _load(a)
    cells = _grid(a)
    print(f"episode {ep.index}: {ep.frames} frames, "
          f"{', '.join(repr(h.name) + (' (from joints)' if h.source == 'fk' else '') for h in hands)}"
          f", {len(cells)} candidate placements at height {a.height:+.3f} m")

    Fs = [feasibility(chain, h.pos, h.quat, cells, a.pos_tol, np.radians(a.rot_tol))
          for h in hands]
    chunks, held = chunk(Fs, cells, window=a.window)

    print(f"\n{len(chunks)} chunk(s); the bases move {len(chunks) - 1} time(s):")
    for i, ch in enumerate(chunks):
        bits = "   ".join(f"{hands[k].name} ({b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f}) "
                          f"held {100 * h:5.1f}%"
                          for k, (b, h) in enumerate(zip(ch.bases, ch.held)))
        print(f"  chunk {i}: frames {ch.start:4d}-{ch.stop:4d} ({len(ch):4d})  {bits}")

    print()
    for k, F in enumerate(Fs):
        cell, share = best_fixed(F, cells)
        print(f"  {hands[k].name:>6}: chunked {100 * held[k].mean():5.1f}% of frames held, against "
              f"{100 * share:5.1f}% for the best single base "
              f"({cell[0]:+.2f}, {cell[1]:+.2f}, {cell[2]:+.2f})")

    if a.out:
        out = dict(dataset=str(a.dataset), episode=int(ep.index), robot=a.robot,
                   window=a.window, hands=[h.name for h in hands],
                   pos_tol=a.pos_tol, rot_tol_deg=a.rot_tol, height=a.height,
                   chunks=[dict(start=ch.start, stop=ch.stop,
                                bases=[[round(float(v), 4) for v in b] for b in ch.bases],
                                held=[round(float(h), 4) for h in ch.held]) for ch in chunks])
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


def cmd_curve(a):
    chain = load_robot(a.robot)
    ep, hands = _load(a)
    cells = _grid(a)
    print(f"episode {ep.index}: how much survives, against how long one base must serve\n")
    print(f"  {'window':>12} {'usable':>8} {'placements':>12}")
    for h in hands:
        F = feasibility(chain, h.pos, h.quat, cells, a.pos_tol, np.radians(a.rot_tol))
        print(f"  {h.name}:")
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
