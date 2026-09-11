"""Offline base-shift probe: does a policy predict the same world-frame motion when the robot stands
somewhere else? No simulator, no new data.

The wrist camera rides on the hand, so a demo frame's image is the same from any base; only the joint
state changes, and OmniBase re-solves it for any base by math. Feed a policy the re-solved state, push
its predicted next joints through FK, and measure the miss against the re-solved target -- in world
centimetres, base-free. A memoriser misses by exactly the shift; a generaliser does not.

    python offline_probe.py E:/data/fastumi/plans2/can_base2.json --out E:/data/baseshift/offline
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "E:/work/Isaacsim/OmniBase")
import numpy as np, omnibase as ob
from omnibase.plan import solve

ap = argparse.ArgumentParser()
ap.add_argument("plan"); ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="2,4,6,8", help="cm, along +-x and +-y")
ap.add_argument("--train-grid", default="-6,-3,0,3,6", help="cm; the multi-base policies train on this x*y grid")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--grasp-window", type=int, default=5, help="rows either side of the jaw-closing row; success is decided there")
a = ap.parse_args()
p = json.load(open(a.plan)); cfg = p["cfg"]; chain = ob.so101()
pos_tol, rot_tol = cfg["pos_tol"], np.radians(cfg["rot_tol"])

# the planned chunks, as world poses
chunks = []
for rec in p["episodes"]:
    if "error" in rec or not rec.get("chunks"): continue
    ep = ob.load(cfg["dataset"], episode=rec["episode"], chain=chain, column=cfg["column"], frame=cfg.get("frame"),
                 level=cfg.get("level"), tcp=cfg.get("tcp"), stride=cfg.get("stride", 1) or 1)
    hand = next(h for h in ep.hands if h.name == cfg["hands"]) if cfg.get("hands") else ep.hands[0]
    for ch in rec["chunks"]:
        s0, s1 = int(ch["start"]), int(ch["stop"]); chunks.append((hand.pos[s0:s1], hand.quat[s0:s1], np.asarray(ch["bases"][0], float), hand.grip[s0:s1] if hand.grip is not None else None))
# the frames that decide a grasp: within --grasp-window of the row the jaws are first told to shut (raw grip < 0);
# a chunk that starts already shut has no grasp in it
W = []
for _, _, _, g in chunks:
    w = np.zeros(len(g) - 1, bool)
    if g is not None and (g < 0).any() and (t := int(np.argmax(g < 0))) > 0: w[max(0, t - a.grasp_window):t + a.grasp_window] = True
    W.append(w)
W = np.concatenate(W)
print(f"{len(chunks)} chunks, {sum(len(c[0]) for c in chunks)} frames")

def resolve(b):
    """(state, action, ok) per chunk at base offset b, joints in radians."""
    out = []
    for pos, quat, base0, _ in chunks:
        q, ok = solve(chain, pos, quat, base0 + b, pos_tol, rot_tol)
        out.append((q[:-1], q[1:], ok[:-1] & ok[1:]))
    return out

def pairs(sol, mask=None):
    S = np.concatenate([s for s, _, _ in sol]); A = np.concatenate([x for _, x, _ in sol]); M = np.concatenate([m for _, _, m in sol])
    return S, A, M if mask is None else (M & mask)

def fk_cm(q):  # tool point in the base frame, cm
    return 100 * chain.fk(q)[0]

# ---- policies: state (N,5) -> next joints (N,5). Trained ones see only base-0 or the grid.
import torch
def mlp_fit(S, A, steps):
    torch.manual_seed(0); mu, sd = S.mean(0), S.std(0) + 1e-6
    X = torch.tensor((S - mu) / sd, dtype=torch.float32); Y = torch.tensor((A - mu) / sd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(5, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 5))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    return lambda Q: net(torch.tensor((Q - mu) / sd, dtype=torch.float32)).detach().numpy() * sd + mu

def knn_fit(S, A):
    return lambda Q: A[np.argmin(((Q[:, None, :] - S[None, :, :]) ** 2).sum(-1), axis=1)]

sol0 = resolve(np.zeros(3)); S0, A0, M0 = pairs(sol0)
grid = [float(v) / 100 for v in a.train_grid.split(",")]
Sg, Ag = [], []
for gx in grid:
    for gy in grid:
        S, A, M = pairs(resolve(np.array([gx, gy, 0.0]))); Sg.append(S[M]); Ag.append(A[M])
Sg, Ag = np.concatenate(Sg), np.concatenate(Ag)
print(f"base-0 pairs {int(M0.sum())}; grid pairs {len(Sg)} over {len(grid)**2} bases")
learned = {"knn (base 0)": knn_fit(S0[M0], A0[M0]), "mlp (base 0)": mlp_fit(S0[M0], A0[M0], a.steps),
           "knn (base grid)": knn_fit(Sg, Ag), "mlp (base grid)": mlp_fit(Sg, Ag, a.steps)}

# ---- the probe
offsets = [("0", np.zeros(3))]
for cm in [float(v) for v in a.offsets.split(",")]:
    for tag, v in (("x+", (1, 0)), ("x-", (-1, 0)), ("y+", (0, 1)), ("y-", (0, -1))):
        offsets.append((f"{tag}{cm:g}", np.array([v[0] * cm / 100, v[1] * cm / 100, 0.0])))
names = ["replay (base-0 rows)", "relative joints (base-0 deltas)"] + list(learned) + ["oracle (re-solved)"]
res = {}
for tag, b in offsets:
    solb = resolve(b); Sb, Ab, Mb = pairs(solb); m = M0 & Mb
    if m.sum() == 0: continue
    target = fk_cm(Ab[m])
    preds = {"replay (base-0 rows)": A0[m], "relative joints (base-0 deltas)": Sb[m] + (A0[m] - S0[m]),
             **{k: f(Sb[m]) for k, f in learned.items()}, "oracle (re-solved)": Ab[m]}
    res[tag] = {k: float(np.median(np.linalg.norm(fk_cm(v) - target, axis=1))) for k, v in preds.items()}
    wm = W[m]; res[tag].update({k + " @grasp": float(np.median(np.linalg.norm(fk_cm(v[wm]) - target[wm], axis=1))) for k, v in preds.items()} if wm.any() else {})
    res[tag]["_grasp_frames"] = int(wm.sum())
    res[tag]["_frames"] = int(m.sum()); res[tag]["_feasible"] = float(Mb.mean())
    print(f"{tag:>5}: {int(m.sum()):3d} frames, {100*Mb.mean():.0f}% feasible  " + "  ".join(f"{k.split(' (')[0]} {v:.1f}" for k, v in res[tag].items() if not k.startswith('_')))

out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "results.json").write_text(json.dumps(res, indent=1))
tags = list(res); lines = ["| policy | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
lines += [f"| {n} | " + " | ".join(f"{res[t][n]:.1f}" for t in tags) + " |" for n in names]
lines += [f"| {n} @grasp | " + " | ".join(f"{res[t].get(n + chr(32) + chr(64) + chr(103) + chr(114) + chr(97) + chr(115) + chr(112), float(chr(110) + chr(97) + chr(110))):.1f}" for t in tags) + " |" for n in names]
lines += ["| grasp-window frames | " + " | ".join(str(res[t]["_grasp_frames"]) for t in tags) + " |"]
lines += ["| frames compared | " + " | ".join(str(res[t]["_frames"]) for t in tags) + " |",
          "| re-solve feasible | " + " | ".join(f"{100*res[t]['_feasible']:.0f}%" for t in tags) + " |"]
(out / "table.md").write_text("median world-frame miss of the predicted next tool point, cm\n\n" + "\n".join(lines) + "\n"); print("\n".join(lines))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, axis in zip(axs, ("x", "y")):
    for n in names:
        pts = sorted((float(t[2:]) * (1 if t[1] == "+" else -1), res[t][n]) for t in tags if t[0] == axis) + [(0.0, res["0"][n])]
        pts.sort(); ax.plot([d for d, _ in pts], [e for _, e in pts], "o-", ms=3, label=n)
    ax.set_xlabel(f"base shift along {axis} (cm)"); ax.grid(alpha=.3)
axs[0].set_ylabel("median miss at the tool point (cm)"); axs[1].legend(fontsize=7); fig.suptitle("Offline base-shift probe (no simulator)"); fig.tight_layout()
fig.savefig(out / "probe.png", dpi=130); print(f"wrote {out}")
