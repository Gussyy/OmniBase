"""Likelihood version of the base-shift probe, on a flow-matching policy, exactly. No simulator.

A flow-matching policy is a continuous normalising flow, so log p(action | state) is exact:
log p1(x1) = log p0(x0) - int_0^1 div v(x_t, t) dt along the ODE through x1. In 5-D the divergence
is five autograd calls, no Hutchinson needed. Two tiny policies are trained on the executable can
chunks (state -> next joints): one at base 0, one on a grid of bases. For every base offset b the
same frames are re-solved by OmniBase and scored:
    NLL(action_b | state_b)   does the policy's distribution move with the base?
    NLL(action_0 | state_b)   does it still believe the base-0 action?
plus the point miss of the sampled mean at the tool point, to tie back to offline_probe.py.

    python flow_probe.py E:/data/fastumi/plans2/can_base2.json --out E:/data/baseshift/flow
"""
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, "E:/work/Isaacsim/OmniBase")
import numpy as np, torch, omnibase as ob
from omnibase.plan import solve

ap = argparse.ArgumentParser()
ap.add_argument("plan"); ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="2,4,6,8"); ap.add_argument("--train-grid", default="-6,-3,0,3,6")
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--ode-steps", type=int, default=40); ap.add_argument("--samples", type=int, default=16)
a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
p = json.load(open(a.plan)); cfg = p["cfg"]; chain = ob.so101(); pos_tol, rot_tol = cfg["pos_tol"], np.radians(cfg["rot_tol"])

chunks = []
for rec in p["episodes"]:
    if "error" in rec or not rec.get("chunks"): continue
    ep = ob.load(cfg["dataset"], episode=rec["episode"], chain=chain, column=cfg["column"], frame=cfg.get("frame"),
                 level=cfg.get("level"), tcp=cfg.get("tcp"), stride=cfg.get("stride", 1) or 1)
    hand = next(h for h in ep.hands if h.name == cfg["hands"]) if cfg.get("hands") else ep.hands[0]
    for ch in rec["chunks"]:
        s0, s1 = int(ch["start"]), int(ch["stop"]); chunks.append((hand.pos[s0:s1], hand.quat[s0:s1], np.asarray(ch["bases"][0], float)))


def resolve(b):
    out = []
    for pos, quat, base0 in chunks:
        q, ok = solve(chain, pos, quat, base0 + b, pos_tol, rot_tol); out.append((q[:-1], q[1:], ok[:-1] & ok[1:]))
    return out


def pairs(sol):
    return (np.concatenate([s for s, _, _ in sol]), np.concatenate([x for _, x, _ in sol]), np.concatenate([m for _, _, m in sol]))


def fk_cm(q): return 100 * chain.fk(q)[0]


class Flow(torch.nn.Module):
    """v(x_t, t | state): the velocity field of a conditional flow from N(0, I) to the action."""
    def __init__(self, d=5, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * d + 1, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, d))

    def forward(self, x, t, s): return self.net(torch.cat([x, s, t], -1))


class Policy:
    def __init__(self, S, A, steps):
        self.mu_s, self.sd_s = S.mean(0), S.std(0) + 1e-6; self.mu_a, self.sd_a = A.mean(0), A.std(0) + 1e-6
        Sn, An = self.ns(S), self.na(A); self.f = Flow(); opt = torch.optim.Adam(self.f.parameters(), 1e-3)
        for _ in range(steps):
            i = torch.randint(0, len(Sn), (512,)); x1 = An[i]; x0 = torch.randn_like(x1); t = torch.rand(512, 1)
            loss = ((self.f((1 - t) * x0 + t * x1, t, Sn[i]) - (x1 - x0)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        self.train_loss = float(loss.detach())

    def ns(self, S): return torch.tensor((S - self.mu_s) / self.sd_s, dtype=torch.float32)
    def na(self, A): return torch.tensor((A - self.mu_a) / self.sd_a, dtype=torch.float32)

    def nll(self, S, A, steps):
        """Exact -log p(A | S) in nats, in the original joint units (the normalisation's Jacobian included)."""
        Sn, x = self.ns(S), self.na(A); dt = 1.0 / steps; logdet = torch.zeros(len(x)); d = x.shape[1]
        for k in range(steps):                       # backwards along the flow, from the data at t=1 to noise at t=0
            t = torch.full((len(x), 1), 1.0 - k * dt); x = x.detach().requires_grad_(True); v = self.f(x, t, Sn)
            div = sum(torch.autograd.grad(v[:, i].sum(), x, retain_graph=i < d - 1)[0][:, i] for i in range(d))
            logdet = logdet + div.detach() * dt; x = x - v.detach() * dt
        x = x.detach(); logp0 = -0.5 * (x ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
        return (-(logp0 - logdet) + float(np.log(self.sd_a).sum())).numpy()

    @torch.no_grad()
    def sample_mean(self, S, K, steps):
        Sn = self.ns(S).repeat_interleave(K, 0); x = torch.randn(len(Sn), 5); dt = 1.0 / steps
        for k in range(steps): x = x + self.f(x, torch.full((len(x), 1), k * dt), Sn) * dt
        return (x.reshape(len(S), K, 5).mean(1).numpy() * self.sd_a + self.mu_a)


sol0 = resolve(np.zeros(3)); S0, A0, M0 = pairs(sol0)
grid = [float(v) / 100 for v in a.train_grid.split(",")]; Sg, Ag = [], []
for gx in grid:
    for gy in grid:
        S, A, M = pairs(resolve(np.array([gx, gy, 0.0]))); Sg.append(S[M]); Ag.append(A[M])
Sg, Ag = np.concatenate(Sg), np.concatenate(Ag)
models = {"flow (base 0)": Policy(S0[M0], A0[M0], a.steps), "flow (base grid)": Policy(Sg, Ag, a.steps)}
print({k: f"train loss {m.train_loss:.3f}" for k, m in models.items()})

offsets = [("0", np.zeros(3))] + [(f"{t}{cm:g}", np.array([v[0] * cm / 100, v[1] * cm / 100, 0.0]))
                                  for cm in [float(v) for v in a.offsets.split(",")] for t, v in (("x+", (1, 0)), ("x-", (-1, 0)), ("y+", (0, 1)), ("y-", (0, -1)))]
res = {}
for tag, b in offsets:
    Sb, Ab, Mb = pairs(resolve(b)); m = M0 & Mb; res[tag] = {"_frames": int(m.sum())}
    for name, pol in models.items():
        nb = pol.nll(Sb[m], Ab[m], a.ode_steps); nr = pol.nll(Sb[m], A0[m], a.ode_steps)
        miss = np.linalg.norm(fk_cm(pol.sample_mean(Sb[m], a.samples, a.ode_steps)) - fk_cm(Ab[m]), axis=1)
        res[tag][name] = dict(nll_resolved=float(np.median(nb)), nll_replay=float(np.median(nr)), miss_cm=float(np.median(miss)))
    print(f"{tag:>4}: " + "   ".join(f"{k.split(' (')[1][:-1]}: NLL re-solved {v['nll_resolved']:.1f} replay {v['nll_replay']:.1f} miss {v['miss_cm']:.1f} cm" for k, v in res[tag].items() if not k.startswith("_")), flush=True)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "results.json").write_text(json.dumps(res, indent=1))
cms = ["0"] + [f"{float(v):g}" for v in a.offsets.split(",")]
def agg(name, key, cm):   # median over the four directions at one radius
    tags = ["0"] if cm == "0" else [t for t in res if t != "0" and t[2:] == cm]
    return float(np.median([res[t][name][key] for t in tags]))
lines = ["| policy | score | " + " | ".join(f"{c} cm" for c in cms) + " |", "|---|---|" + "---|" * len(cms)]
for name in models:
    for key, label in (("nll_resolved", "NLL of the re-solved action (nats)"), ("nll_replay", "NLL of the base-0 action (nats)"), ("miss_cm", "miss of the sampled mean (cm)")):
        lines.append(f"| {name} | {label} | " + " | ".join(f"{agg(name, key, c):.1f}" for c in cms) + " |")
md = "median over the four shift directions; NLL in nats per 5-D action, same units for both policies\n\n" + "\n".join(lines)
(out / "table.md").write_text(md + "\n"); print(md)
