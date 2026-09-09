"""Fetch FastUMI-100K from the Hub in the order that makes the budget go furthest.

The point of this script is what it does NOT download. A FastUMI episode is 30 kB of poses and
10 MB of video, and every question OmniBase asks -- where would a base have to stand, how much of
this episode could an SO-101 hold, is it worth having -- is answered by the 30 kB. So take the
poses for everything, decide, and then fetch video only for the episodes that survived.

    python scripts/fetch_fastumi.py meta   E:/data/fastumi put_shoes_into_storage_box ...
    python -m omnibase sweep  ...                      # per task, using the poses
    python -m omnibase select ... --budget-gb 90 --out selection.json
    python scripts/fetch_fastumi.py videos E:/data/fastumi selection.json --workers 12

``meta`` also writes ``sizes.json`` per task -- the real byte size of every episode's video,
straight from the Hub -- so the selection weighs real bytes rather than an average.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO = "IPEC-COMMUNITY/FastUMI_100k_lerobot"
CAMERA = "observation.images.camera_rgb_image"


def video_path(task, episode, chunk=1000):
    return (f"single_arm/{task}/videos/chunk-{episode // chunk:03d}/{CAMERA}/"
            f"episode_{episode:06d}.mp4")


def cmd_meta(a):
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for task in a.tasks:
        t0 = time.time()
        snapshot_download(a.repo, repo_type="dataset", local_dir=a.root, max_workers=a.workers,
                          allow_patterns=[f"single_arm/{task}/meta/*",
                                          f"single_arm/{task}/data/*"])
        sizes = {}
        for f in api.list_repo_tree(a.repo, path_in_repo=f"single_arm/{task}/videos",
                                    recursive=True, repo_type="dataset"):
            if getattr(f, "size", None) and f.path.endswith(".mp4"):
                sizes[int(f.path.rsplit("episode_", 1)[1][:6])] = f.size
        out = Path(a.root) / "single_arm" / task / "sizes.json"
        out.write_text(json.dumps(sizes), encoding="utf-8")
        print(f"{task}: poses in, {len(sizes)} videos listed totalling "
              f"{sum(sizes.values()) / 1e9:.1f} GB (not downloaded), {time.time() - t0:.0f}s",
              flush=True)


def cmd_videos(a):
    from huggingface_hub import hf_hub_download

    chosen = json.loads(Path(a.selection).read_text(encoding="utf-8"))["chosen"]
    jobs = []
    for rec in chosen:
        task = Path(rec["dataset"]).name
        rel = video_path(task, rec["episode"])
        if not (Path(a.root) / rel).exists():
            jobs.append((rel, rec.get("bytes", 10e6)))
    total = sum(b for _, b in jobs) / 1e9
    print(f"{len(chosen)} episodes selected, {len(jobs)} still to fetch, {total:.1f} GB",
          flush=True)
    if a.dry_run or not jobs:
        return

    t0, done = time.time(), [0]

    def get(job):
        rel, size = job
        for attempt in range(3):
            try:
                hf_hub_download(a.repo, rel, repo_type="dataset", local_dir=a.root)
                break
            except Exception as exc:                      # noqa: BLE001 -- retried, then reported
                if attempt == 2:
                    print(f"\n  failed: {rel}: {exc}", flush=True)
                time.sleep(2 * (attempt + 1))
        done[0] += 1
        if done[0] % 25 == 0:
            dt = time.time() - t0
            rate = done[0] / max(dt, 1e-9)
            print(f"\r  {done[0]}/{len(jobs)}  {rate * 3600:.0f}/hour, "
                  f"{(len(jobs) - done[0]) / max(rate, 1e-9) / 60:.0f} min left", end="",
                  flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(get, jobs))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser("fetch_fastumi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=REPO)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("meta", help="poses and metadata for whole tasks, plus the video sizes")
    q.add_argument("root")
    q.add_argument("tasks", nargs="+")
    q.add_argument("--workers", type=int, default=16)
    q.set_defaults(func=cmd_meta)

    q = sub.add_parser("videos", help="video for the episodes a selection kept")
    q.add_argument("root")
    q.add_argument("selection")
    q.add_argument("--workers", type=int, default=12)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_videos)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
