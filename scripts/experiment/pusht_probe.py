"""Does an offline score relate to success? PushT + lerobot/diffusion_pusht, per episode.

For each demonstration episode: (a) roll the policy out from that episode's own initial state and
record success and best coverage; (b) score the demonstration offline under the same policy --
the diffusion loss on its chunks, the distance of its chunks to what the policy would sample, the
spread of those samples, and how much the sampled chunk moves when the state is perturbed with the
image unchanged (state reliance). Then correlate (b) with (a) across episodes.

    python pusht_probe.py --out E:/data/pusht/probe [--limit 8] [--every 5] [--envs 32]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, zarr, gymnasium as gym, gym_pusht  # noqa: F401
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--every", type=int, default=5, help="score every Nth frame of an episode"); ap.add_argument("--envs", type=int, default=32)
ap.add_argument("--samples", type=int, default=8); ap.add_argument("--noise-draws", type=int, default=4)
ap.add_argument("--delta", type=float, default=20.0, help="px, state perturbation")
ap.add_argument("--seeds", type=int, default=1, help="rollouts per episode (policy noise)")
ap.add_argument("--random", type=int, default=0, help="sanity check only: N rollouts from random starts, then exit")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

pol = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(dev).eval(); c = pol.config
dt = {k: [i / 10 for i in idx] for k, idx in (("observation.image", c.observation_delta_indices),
                                             ("observation.state", c.observation_delta_indices),
                                             ("action", c.action_delta_indices))}
ds = LeRobotDataset("lerobot/pusht", delta_timestamps=dt)
# The checkpoint predates LeRobot's processor pipeline, so its normalisation buffers are rejected on load. Rebuild the
# pipeline from THOSE buffers, not the dataset stats: it normalised images with ImageNet mean/std (0.485/0.229...),
# and the dataset's own image stats (mean 0.97, the white board) make it score 1/32 instead of the reported 65%.
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
sd = load_file(hf_hub_download("lerobot/diffusion_pusht", "model.safetensors")); stats = {k: dict(v) for k, v in ds.meta.stats.items()}
for k, v in sd.items():
    if k.startswith("normalize_inputs.buffer_") or k.startswith("normalize_targets.buffer_"):
        key, stat = k.split("buffer_")[1].rsplit(".", 1); key = key.replace("observation_", "observation."); stats.setdefault(key, {})[stat] = v.numpy()
pre, post = make_pre_post_processors(c, dataset_stats=stats)
z = zarr.open("E:/data/pusht/pusht/pusht_cchi_v7_replay.zarr", mode="r")
Z = z["data/state"][:]; ends = z["meta/episode_ends"][:]; starts = np.r_[0, ends[:-1]]
eps = list(range(ds.num_episodes))[: a.limit]
for e in eps[:3]:   # the zarr and the LeRobot dataset must be the same episodes in the same order
    f = ds.meta.episodes[e]["dataset_from_index"]
    assert np.allclose(Z[starts[e], :2], ds[f]["observation.state"][-1].numpy(), atol=1.0), (e, Z[starts[e], :2], ds[f]["observation.state"][-1])
print(f"{len(eps)} episodes; zarr aligned", flush=True)


def collate(idx):
    items = [ds[i] for i in idx]
    return {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k]) and items[0][k].dim() > 0}


def stack_images(b):
    b["observation.images"] = torch.stack([b[k] for k in c.image_features], dim=-4); return b


@torch.no_grad()
def per_sample_loss(b, draws):
    """The policy's own denoising loss, per sample, averaged over `draws` noise draws (masked like training)."""
    m = pol.diffusion; b = stack_images(dict(b)); cond = m._prepare_global_conditioning(b); traj = b["action"]; tot = 0
    for _ in range(draws):
        eps_ = torch.randn_like(traj)
        t = torch.randint(0, m.noise_scheduler.config.num_train_timesteps, (traj.shape[0],), device=dev)
        pred = m.unet(m.noise_scheduler.add_noise(traj, eps_, t), t, global_cond=cond)
        target = eps_ if c.prediction_type == "epsilon" else traj
        l = (pred - target) ** 2; mask = (~b["action_is_pad"]).unsqueeze(-1).float()
        tot = tot + (l * mask).sum((1, 2)) / (mask.sum((1, 2)) * l.shape[-1]).clamp_min(1)
    return (tot / draws).cpu().numpy()


@torch.no_grad()
def sample_chunks(b, K):
    """K sampled executed chunks (B, K, n_action_steps, 2) in pixels."""
    obs = {k: b[k].repeat_interleave(K, 0) for k in ("observation.image", "observation.state")}
    act = post(pol.predict_action_chunk(stack_images(obs)))
    return act.reshape(b["observation.state"].shape[0], K, *act.shape[1:]).cpu().numpy()


# ---- (a) rollouts from each demo's initial state, batched across envs
def rollout(episodes, seed):
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300) for _ in episodes]
    hist = []; res = {e: dict(success=False, coverage=0.0, steps=0) for e in episodes}
    for env, e in zip(envs, episodes):
        env.reset(seed=seed); o, _ = env.reset(options={"reset_to_state": Z[starts[e]].astype(np.float64)}); hist.append([o, o])
    alive = list(range(len(episodes))); torch.manual_seed(seed)
    for step in range(0, 300, c.n_action_steps):
        if not alive: break
        img = torch.stack([torch.stack([torch.from_numpy(h["pixels"]).permute(2, 0, 1).float() / 255 for h in hist[i]]) for i in alive])
        st = torch.stack([torch.stack([torch.from_numpy(h["agent_pos"]).float() for h in hist[i]]) for i in alive])
        b = pre({"observation.image": img, "observation.state": st})
        acts = post(pol.predict_action_chunk(stack_images(b))).cpu().numpy()
        still = []
        for j, i in enumerate(alive):
            done = False
            for k in range(c.n_action_steps):
                o, r, term, trunc, info = envs[i].step(acts[j, k]); e = episodes[i]; res[e]["steps"] += 1
                res[e]["coverage"] = max(res[e]["coverage"], float(r)); res[e]["success"] |= bool(info.get("is_success", False))
                hist[i] = [hist[i][1], o]
                if term or trunc:
                    done = True; break
            if not done: still.append(i)
        alive = still
    for env in envs: env.close()
    return res


if a.random:   # sanity check of the rollout loop against the checkpoint's reported 65.4%: random starts, as lerobot-eval does
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300) for _ in range(a.random)]
    for i, env in enumerate(envs): env.reset(seed=i)
    Z = np.stack([np.r_[env.unwrapped.agent.position, env.unwrapped.block.position, env.unwrapped.block.angle] for env in envs]); starts = np.arange(a.random)
    for env in envs: env.close()
    r = rollout(list(range(a.random)), seed=0)
    print(f"random starts: {sum(v['success'] for v in r.values())}/{a.random} succeeded, mean best coverage {np.mean([v['coverage'] for v in r.values()]):.2f}"); raise SystemExit

t0 = time.time(); roll = {e: [] for e in eps}
for s in range(a.seeds):
    for i in range(0, len(eps), a.envs):
        r = rollout(eps[i:i + a.envs], seed=1000 * s + i)
        for e, v in r.items(): roll[e].append(v)
    n_ok = sum(any(v["success"] for v in roll[e]) for e in eps)
    print(f"seed {s}: {n_ok}/{len(eps)} episodes succeeded at least once  ({time.time()-t0:.0f}s)", flush=True)

# ---- (b) offline scores on the demonstrations
rows = []; t0 = time.time()
for n, e in enumerate(eps):
    m = ds.meta.episodes[e]; idx = list(range(m["dataset_from_index"], m["dataset_to_index"], a.every)); b = collate(idx)
    gt = b["action"][:, c.n_obs_steps - 1: c.n_obs_steps - 1 + c.n_action_steps].numpy()   # the executed part of the chunk, px
    nb = pre(dict(b)); loss = per_sample_loss(nb, a.noise_draws)
    S = sample_chunks(nb, a.samples)                                                      # (F, K, 8, 2) px
    d = np.linalg.norm(S - gt[:, None], axis=-1).mean(-1)                                 # (F, K) mean px error per sample
    spread = np.linalg.norm(S[:, :, None] - S[:, None, :], axis=-1).mean((-1, -2, -3))    # (F,)
    st2 = b["observation.state"].clone(); st2[..., 0] += a.delta
    nb2 = dict(nb); nb2["observation.state"] = pre({"observation.state": st2, "observation.image": b["observation.image"]})["observation.state"]
    S2 = sample_chunks(nb2, a.samples); shift = np.linalg.norm(S2.mean(1) - S.mean(1), axis=-1).mean(-1)   # (F,) px the chunk moved
    rows.append(dict(episode=e, frames=len(idx), length=m["length"],
                     success=float(np.mean([v["success"] for v in roll[e]])), coverage=float(np.mean([v["coverage"] for v in roll[e]])),
                     steps=float(np.mean([v["steps"] for v in roll[e]])),
                     loss=float(loss.mean()), loss_p90=float(np.percentile(loss, 90)), min_dist=float(d.min(1).mean()),
                     mean_dist=float(d.mean()), spread=float(spread.mean()), state_reliance=float(shift.mean() / a.delta)))
    if n % 20 == 0: print(f"  scored {n+1}/{len(eps)} ({time.time()-t0:.0f}s): {rows[-1]}", flush=True)
(out / "episodes.json").write_text(json.dumps(rows, indent=1))

# ---- relate
from scipy.stats import spearmanr, mannwhitneyu
succ = np.array([r["success"] >= 0.5 for r in rows]); cov = np.array([r["coverage"] for r in rows])
lines = [f"{len(rows)} episodes, {a.seeds} rollout(s) each: success {100*succ.mean():.0f}% (reported for this checkpoint: 65.4%), "
         f"mean best coverage {cov.mean():.2f}", "",
         "| offline score | AUC for success | Spearman vs coverage | mean (success) | mean (fail) |", "|---|---|---|---|---|"]
for k in ("loss", "loss_p90", "min_dist", "mean_dist", "spread", "state_reliance"):
    x = np.array([r[k] for r in rows]); auc = float("nan")
    if succ.any() and (~succ).any():
        u = mannwhitneyu(x[succ], x[~succ]).statistic; auc = u / (succ.sum() * (~succ).sum())
    rho = spearmanr(x, cov).correlation
    mf = x[~succ].mean() if (~succ).any() else float("nan")
    lines.append(f"| {k} | {auc:.2f} | {rho:+.2f} | {x[succ].mean():.3g} | {mf:.3g} |")
lines += ["", "AUC 0.5 = the score says nothing about success; below 0.5 = higher score, less success (as a loss should)."]
(out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, k in zip(axs, ("loss", "min_dist", "spread", "state_reliance")):
    x = np.array([r[k] for r in rows]); ax.scatter(x, cov, c=succ, cmap="coolwarm", s=14); ax.set_xlabel(k); ax.grid(alpha=.3)
axs[0].set_ylabel("best coverage in rollout"); fig.suptitle("PushT: offline scores of a demo vs the policy's success from its start")
fig.tight_layout(); fig.savefig(out / "scores.png", dpi=130); print(f"wrote {out}")
