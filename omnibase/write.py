"""Writing a LeRobot dataset, so a plan can be trained on rather than only read.

Everything else in this library answers a question about a recording. This turns the answer
into the thing a policy actually consumes: joint angles for a real arm, the wrist video cut to
match them, and the metadata LeRobot needs to load it without conversion.

The layout is LeRobotDataset **v3.0**, written directly rather than through LeRobot itself.
That is not stubbornness -- LeRobot pins torch and a large dependency tree, and this library is
numpy and scipy. The format is a handful of parquet files and some JSON; importing a deep
learning framework to produce them would cost more than writing them.

    <root>/
      meta/info.json                             feature schema, counts, path templates
      meta/stats.json                            what a trainer normalises with
      meta/tasks.parquet                         language string -> task index
      meta/episodes/chunk-000/file-000.parquet   one row per episode: its slice, its video
      data/chunk-000/file-000.parquet            every episode's rows, concatenated
      videos/<key>/chunk-000/file-000.mp4        one clip per episode

One video file per episode, each starting at timestamp zero. v3.0 allows several episodes to
share a file and be found by ``from_timestamp``, which is what a recorder writing a long session
should do; here every episode is a slice cut out of somebody else's video, and stapling them
back together only to record where the seams are would be work in exchange for a way to be
wrong.

ffmpeg is called as a subprocess. It is the one external dependency, it is already required to
read what LeRobot writes, and it slices and rescales without this process ever holding a frame.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

#: LeRobot's own file-size ceilings, in MB. Written into ``info.json`` because it validates them
#: as positive; not enforced here, because one episode per file never approaches them.
DATA_FILE_MB = 100
VIDEO_FILE_MB = 200
#: Episodes per chunk directory, LeRobot's default.
CHUNK = 1000


def ffmpeg(args, what):
    """Run ffmpeg, and say what was being done if it fails."""
    done = subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to {what}: {done.stderr.strip()[-700:]}")


def frame_count(path):
    """Frames actually in a video file. The check that a slice lines up with its rows."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return -1


def slice_video(src, dst, start, stop, stride, fps, size, crf=23, offset=0):
    """Cut ``[start, stop)`` of ``src``, every ``stride``-th frame, to ``dst`` at ``size``.

    Selected by frame NUMBER rather than by timestamp. The rows being written are frame indices
    into the source parquet, and matching them to the video by seconds means trusting two clocks
    to agree; matching them by count means trusting one. ``-bf 0`` keeps the first frame at
    presentation time zero, which a B-frame reordering offset would quietly break.
    """
    w, h = size
    sel = f"between(n\\,{start + offset}\\,{stop - 1 + offset})"
    if stride > 1:
        sel += f"*not(mod(n-{start + offset}\\,{stride}))"
    vf = (f"select='{sel}',setpts=N/{fps}/TB,"
          f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:-1:-1:color=black")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-i", str(src), "-vf", vf, "-fps_mode", "passthrough", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            "-preset", "veryfast", "-g", "10", "-bf", "0", str(dst)],
           f"slice frames {start}-{stop} of {Path(src).name}")


def first_frame(path, size, lit=18):
    """One decoded frame: channel statistics, and where the picture actually is.

    LeRobot refuses to normalise a camera it has no statistics for, and computing them properly
    means decoding every frame of every episode. One frame per episode, averaged over thousands
    of episodes, is within a per cent of that and costs nothing.

    The second thing is the bounding box of the non-black pixels. A fisheye lens does not fill
    its sensor -- these recordings are a bright ellipse on a black rectangle -- and a policy
    trained on that will see black corners at deployment or it will not recognise the view.
    Recording where the picture sits means the simulator can be masked to match, rather than
    somebody eyeballing it later.
    """
    w, h = size
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True)
    px = np.frombuffer(out.stdout, dtype=np.uint8)
    if px.size < h * w * 3:
        return None
    img = px[: h * w * 3].reshape(h, w, 3)
    flat = img.reshape(-1, 3).astype(np.float64) / 255.0
    ys, xs = np.nonzero(img.max(axis=2) > lit)
    box = None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return dict(mean=flat.mean(0), sq=(flat * flat).mean(0), min=flat.min(0), max=flat.max(0),
                box=box)


class Writer:
    """Collects episodes, then writes the dataset. Rows are held in memory; videos are not.

    Args:
        root: output directory. Refused if a dataset already lives there.
        fps: the rate the rows are at, after any subsampling.
        names: one name per state/action element.
        video_key: the camera feature name, e.g. ``observation.images.wrist``.
        size: ``(width, height)`` the clips were scaled to.
        robot_type: written into ``info.json`` for whoever reads it later.
    """

    def __init__(self, root, fps, names, video_key="observation.images.wrist",
                 size=(512, 384), robot_type="so101"):
        self.root = Path(root)
        if (self.root / "meta" / "info.json").exists():
            raise FileExistsError(f"{self.root} already holds a dataset; point --out elsewhere")
        self.fps, self.names = int(fps), list(names)
        self.video_key, self.size, self.robot_type = video_key, tuple(size), robot_type
        self.episodes, self.tasks, self.frames = [], {}, 0
        self._acc = {}
        self._box = None

    def add(self, state, action, task, video, extra=None):
        """One episode: ``(n, d)`` states and actions, a language string, a finished clip.

        The clip is moved into place here rather than copied, because it was cut straight to a
        staging path and nothing else refers to it.
        """
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        n = len(state)
        if n == 0 or len(action) != n:
            raise ValueError(f"{n} states and {len(action)} actions do not make an episode")
        idx = len(self.episodes)
        dst = self.root / f"videos/{self.video_key}/chunk-{idx // CHUNK:03d}/file-{idx % CHUNK:03d}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(video), dst)

        self.tasks.setdefault(task, len(self.tasks))
        row = dict(episode_index=idx, tasks=[task], length=n,
                   state=state, action=action,
                   task_index=self.tasks[task],
                   dataset_from_index=self.frames, dataset_to_index=self.frames + n)
        row.update(extra or {})
        self.episodes.append(row)
        self.frames += n
        self._fold("observation.state", state.astype(np.float64))
        self._fold("action", action.astype(np.float64))
        img = first_frame(dst, self.size)
        if img:
            st = self._acc.setdefault(self.video_key, dict(n=0, sum=0.0, sq=0.0,
                                                           min=None, max=None))
            st["n"] += 1
            st["sum"] = st["sum"] + img["mean"]
            st["sq"] = st["sq"] + img["sq"]
            st["min"] = img["min"] if st["min"] is None else np.minimum(st["min"], img["min"])
            st["max"] = img["max"] if st["max"] is None else np.maximum(st["max"], img["max"])
            if img.get("box"):
                b = img["box"]
                self._box = b if self._box is None else (
                    min(self._box[0], b[0]), min(self._box[1], b[1]),
                    max(self._box[2], b[2]), max(self._box[3], b[3]))
        return idx

    def _fold(self, key, a):
        st = self._acc.setdefault(key, dict(n=0, sum=0.0, sq=0.0, min=None, max=None))
        st["n"] += len(a)
        st["sum"] = st["sum"] + a.sum(0)
        st["sq"] = st["sq"] + (a * a).sum(0)
        st["min"] = a.min(0) if st["min"] is None else np.minimum(st["min"], a.min(0))
        st["max"] = a.max(0) if st["max"] is None else np.maximum(st["max"], a.max(0))

    def duplicate(self, idx, times):
        """Repeat an episode ``times`` extra times, video and all.

        Sixteen episodes of the task you actually care about, mixed into ten thousand of
        something else, are 0.2% of the sampler's attention. Copying them is the whole of what
        an episode weight would do here, LeRobot having none, and it is four lines.
        """
        src = self.episodes[idx]
        vid = self.root / f"videos/{self.video_key}/chunk-{idx // CHUNK:03d}/file-{idx % CHUNK:03d}.mp4"
        for _ in range(int(times)):
            new = len(self.episodes)
            dst = self.root / f"videos/{self.video_key}/chunk-{new // CHUNK:03d}/file-{new % CHUNK:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(vid, dst)
            row = dict(src)
            row.update(episode_index=new, dataset_from_index=self.frames,
                       dataset_to_index=self.frames + src["length"])
            self.episodes.append(row)
            self.frames += src["length"]
            self._fold("observation.state", src["state"].astype(np.float64))
            self._fold("action", src["action"].astype(np.float64))

    # ---- the files ---------------------------------------------------------------------

    def _features(self):
        w, h = self.size
        d = len(self.names)
        feats = {
            "observation.state": dict(dtype="float32", shape=[d], names=self.names, fps=self.fps),
            "action": dict(dtype="float32", shape=[d], names=self.names, fps=self.fps),
            "timestamp": dict(dtype="float32", shape=[1], names=None, fps=self.fps),
            "frame_index": dict(dtype="int64", shape=[1], names=None, fps=self.fps),
            "episode_index": dict(dtype="int64", shape=[1], names=None, fps=self.fps),
            "index": dict(dtype="int64", shape=[1], names=None, fps=self.fps),
            "task_index": dict(dtype="int64", shape=[1], names=None, fps=self.fps),
        }
        feats[self.video_key] = dict(
            dtype="video", shape=[h, w, 3], names=["height", "width", "channel"],
            info={"video.fps": float(self.fps), "video.codec": "h264",
                  "video.pix_fmt": "yuv420p", "video.height": h, "video.width": w,
                  "video.channels": 3, "video.is_depth_map": False, "has_audio": False})
        return feats

    def _stats(self):
        out = {}
        for key, st in self._acc.items():
            n = max(1, st["n"])
            mean = np.asarray(st["sum"]) / n
            std = np.sqrt(np.maximum(np.asarray(st["sq"]) / n - mean * mean, 0.0))
            vals = dict(mean=mean, std=std, min=st["min"], max=st["max"])
            if key == self.video_key:
                vals = {k: np.asarray(v).reshape(3, 1, 1) for k, v in vals.items()}
            out[key] = {k: np.asarray(v).tolist() for k, v in vals.items()}
            out[key]["count"] = [self.frames if key != self.video_key else st["n"]]
        return out

    def finish(self):
        """Write the dataset. Returns the root."""
        import pandas as pd

        if not self.episodes:
            raise ValueError("no episodes were added; nothing to write")
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

        # data: one file per chunk of episodes, with a global row index that never restarts.
        rows_by_chunk = {}
        for ep in self.episodes:
            c = ep["episode_index"] // CHUNK
            n = ep["length"]
            rows_by_chunk.setdefault(c, []).append(pd.DataFrame({
                "observation.state": list(ep["state"]),
                "action": list(ep["action"]),
                "timestamp": (np.arange(n) / self.fps).astype(np.float32),
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, ep["episode_index"], dtype=np.int64),
                "index": np.arange(ep["dataset_from_index"], ep["dataset_to_index"],
                                   dtype=np.int64),
                "task_index": np.full(n, ep["task_index"], dtype=np.int64),
            }))
        for c, parts in rows_by_chunk.items():
            f = self.root / f"data/chunk-{c:03d}/file-000.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(f, index=False)

        # meta/episodes: where each episode's rows and video live.
        meta = []
        for ep in self.episodes:
            i = ep["episode_index"]
            meta.append({
                "episode_index": i, "tasks": ep["tasks"], "length": ep["length"],
                "data/chunk_index": i // CHUNK, "data/file_index": 0,
                "dataset_from_index": ep["dataset_from_index"],
                "dataset_to_index": ep["dataset_to_index"],
                "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                f"videos/{self.video_key}/chunk_index": i // CHUNK,
                f"videos/{self.video_key}/file_index": i % CHUNK,
                f"videos/{self.video_key}/from_timestamp": 0.0,
                f"videos/{self.video_key}/to_timestamp": ep["length"] / self.fps,
                **{k: v for k, v in ep.items()
                   if k not in ("episode_index", "tasks", "length", "state", "action",
                                "task_index", "dataset_from_index", "dataset_to_index")},
            })
        f = self.root / "meta/episodes/chunk-000/file-000.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta).to_parquet(f, index=False)

        # meta/tasks: read back by position, so the rows must be in index order.
        names = sorted(self.tasks, key=self.tasks.get)
        pd.DataFrame({"task_index": [self.tasks[t] for t in names]},
                     index=pd.Index(names, name="task")).to_parquet(self.root / "meta/tasks.parquet")

        (self.root / "meta/stats.json").write_text(json.dumps(self._stats(), indent=1),
                                                   encoding="utf-8")
        info = dict(codebase_version="v3.0", robot_type=self.robot_type, fps=self.fps,
                    total_episodes=len(self.episodes), total_frames=self.frames,
                    total_tasks=len(self.tasks), chunks_size=CHUNK,
                    data_files_size_in_mb=DATA_FILE_MB, video_files_size_in_mb=VIDEO_FILE_MB,
                    splits={"train": f"0:{len(self.episodes)}"},
                    data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    features=self._features())
        (self.root / "meta/info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

        # Beside LeRobot's own metadata, not inside it: what a reader needs to reproduce this
        # dataset's view at deployment. LeRobot warns about keys it does not know, and it is
        # right to -- this is OmniBase's business, not the format's.
        extra = dict(camera=self.video_key, size=list(self.size))
        if self._box:
            x0, y0, x1, y1 = self._box
            w, h = self.size
            extra["ellipse"] = [round((x0 + x1) / 2 / w, 4), round((y0 + y1) / 2 / h, 4),
                                round((x1 - x0) / 2 / w, 4), round((y1 - y0) / 2 / h, 4)]
            extra["ellipse_note"] = ("centre and radii as fractions of the frame: where the "
                                     "lens's picture actually is. Mask a rendered camera to "
                                     "this before showing it to a policy trained here.")
        (self.root / "meta/omnibase.json").write_text(json.dumps(extra, indent=1),
                                                      encoding="utf-8")
        return self.root
