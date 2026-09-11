"""Action ambiguity under base augmentation: the miss a policy incurs if it cannot tell the bases
apart and averages their actions. Data only, no training, no simulator.

Multi-base augmentation writes the same demo from several bases. The wrist image is identical from
all of them, so an image-dominant policy sees one observation with several joint-space labels and
learns their mean. Pushing that mean through FK from each base says, in centimetres, how far off the
tool lands. For an action space that is base-invariant (relative EE) the number is zero by construction.

    python ambiguity.py E:/data/fastumi/plans2/can_base2.json --spans 0,2,4,6,8,12
"""
import argparse, json, sys
sys.path.insert(0, "E:/work/Isaacsim/OmniBase")
import numpy as np, omnibase as ob
from omnibase.plan import solve

ap = argparse.ArgumentParser(); ap.add_argument("plan"); ap.add_argument("--spans", default="0,2,4,6,8,12", help="cm: half-width of a 3x3 base grid")
ap.add_argument("--zspan", type=float, default=0.0, help="cm: also +- this in z (3x3x3 grid)")
a = ap.parse_args()
p = json.load(open(a.plan)); cfg = p["cfg"]; chain = ob.so101(); pos_tol, rot_tol = cfg["pos_tol"], np.radians(cfg["rot_tol"])
chunks = []
for rec in p["episodes"]:
    if "error" in rec or not rec.get("chunks"): continue
    ep = ob.load(cfg["dataset"], episode=rec["episode"], chain=chain, column=cfg["column"], frame=cfg.get("frame"), level=cfg.get("level"), tcp=cfg.get("tcp"), stride=cfg.get("stride", 1) or 1)
    hand = next(h for h in ep.hands if h.name == cfg["hands"]) if cfg.get("hands") else ep.hands[0]
    for ch in rec["chunks"]:
        s0, s1 = int(ch["start"]), int(ch["stop"]); chunks.append((hand.pos[s0:s1], hand.quat[s0:s1], np.asarray(ch["bases"][0], float)))

print("| base spread (3x3 grid, +- cm) | bases | frames feasible from all | absolute joints: miss (cm) | joint deltas: miss (cm) | relative EE |")
print("|---|---|---|---|---|---|")
for span in [float(v) for v in a.spans.split(",")]:
    s = span / 100; zs = [0.0] if not a.zspan else [-a.zspan / 100, 0.0, a.zspan / 100]
    offs = [np.array([x, y, z]) for x in (-s, 0, s) for y in (-s, 0, s) for z in zs] if span else [np.zeros(3)]
    miss_abs, miss_del, n = [], [], 0
    for pos, quat, base0 in chunks:
        sols = [solve(chain, pos, quat, base0 + o, pos_tol, rot_tol) for o in offs]
        ok = np.all([k[:-1] & k[1:] for _, k in sols], axis=0)      # feasible from every base, this and next frame
        if not ok.any(): continue
        S = np.stack([q[:-1] for q, _ in sols]); A = np.stack([q[1:] for q, _ in sols])   # (B, T, 5)
        mean_abs = A.mean(0); mean_del = (A - S).mean(0)
        for bi in range(len(offs)):
            target = chain.fk(A[bi][ok])[0]
            miss_abs.append(np.linalg.norm(chain.fk(mean_abs[ok])[0] - target, axis=1))
            miss_del.append(np.linalg.norm(chain.fk(S[bi][ok] + mean_del[ok])[0] - target, axis=1))
        n += int(ok.sum())
    if not miss_abs:
        print(f"| {span:g} | {len(offs)} | 0 | - | - | 0 |"); continue
    ma, md = np.concatenate(miss_abs) * 100, np.concatenate(miss_del) * 100
    print(f"| {span:g} | {len(offs)} | {n} | {np.median(ma):.1f} (p90 {np.percentile(ma, 90):.1f}) | {np.median(md):.1f} (p90 {np.percentile(md, 90):.1f}) | 0 |")
