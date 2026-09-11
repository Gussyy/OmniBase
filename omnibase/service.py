"""The command line as a web service, with a page in front of it.

    pip install 'omnibase[service]'
    omnibase serve --host 0.0.0.0 --port 8000

Every job is one CLI invocation run as a subprocess, so the service adds nothing the command
line cannot do and the log you see is the output you would have seen. Jobs run one at a time
-- a sweep already uses every core through ``--workers``. Datasets are read from paths on the
machine the service runs on (mount them into the container); there is no upload and no
authentication, so put it behind whatever your network already trusts.

    POST /jobs   {"tool": "place", "args": {"dataset": "...", "hands": "right", "step": 0.05}}
    GET  /jobs, /jobs/{id}, /jobs/{id}/result, /jobs/{id}/log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

WORK = Path(os.environ.get("OMNIBASE_WORK", "omnibase_jobs")).resolve()
TOOLS = {"place", "report", "ambiguity", "probe", "sweep", "data"}
#: the argument the tool's result file is written through, and what it is
RESULT = {"place": ("place-out", "json"), "report": ("out", "md"), "ambiguity": ("out", "json"),
          "probe": ("out", "json"), "sweep": ("out", "json"), "data": (None, None)}

app = FastAPI(title="OmniBase", description=__doc__)
JOBS: dict[str, dict] = {}
_queue: list[str] = []
_lock = threading.Lock()


class Job(BaseModel):
    tool: str
    args: dict = {}


def _argv(tool, args, result_path):
    """A CLI argv from a JSON object: positional keys first, then ``--key value`` for the rest."""
    positional = {"place": ["dataset"], "sweep": ["dataset"], "data": ["dataset"],
                  "report": ["plans"], "ambiguity": ["plans"], "probe": ["plans"]}[tool]
    argv = [sys.executable, "-m", "omnibase", tool]
    for k in positional:
        if k not in args or args[k] in ("", None):
            raise HTTPException(400, f"{tool} needs {k!r}")
        argv.append(str(args[k]))
    for k, v in args.items():
        if k in positional or v in ("", None, False):
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif isinstance(v, list):
            for item in v:
                argv += [flag, str(item)]
        else:
            argv += [flag + "=" + str(v)] if str(v).startswith("-") else [flag, str(v)]
    out_flag, _ = RESULT[tool]
    if out_flag and f"--{out_flag}" not in argv:
        argv += [f"--{out_flag}", str(result_path)]
    return argv


def _run(job_id):
    j = JOBS[job_id]
    j["status"], j["started"] = "running", time.time()
    log = Path(j["log_path"])
    with log.open("w", encoding="utf-8") as fh:
        try:
            p = subprocess.run(j["argv"], stdout=fh, stderr=subprocess.STDOUT, cwd=str(WORK), timeout=None)
            j["returncode"] = p.returncode
            j["status"] = "done" if p.returncode == 0 else "failed"
        except Exception as exc:                        # noqa: BLE001 -- the job's error, not the service's
            j["status"], j["error"] = "failed", f"{type(exc).__name__}: {exc}"
    j["finished"] = time.time()


def _worker():
    while True:
        with _lock:
            job_id = _queue.pop(0) if _queue else None
        if job_id is None:
            time.sleep(0.5); continue
        _run(job_id)


threading.Thread(target=_worker, daemon=True).start()    # ponytail: one worker, jobs in order


@app.post("/jobs")
def create(job: Job):
    if job.tool not in TOOLS:
        raise HTTPException(400, f"tool must be one of {sorted(TOOLS)}")
    WORK.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    _, kind = RESULT[job.tool]
    result_path = WORK / f"{job_id}.{kind}" if kind else None
    argv = _argv(job.tool, job.args, result_path)
    JOBS[job_id] = dict(id=job_id, tool=job.tool, args=job.args, argv=argv, status="queued",
                        created=time.time(), log_path=str(WORK / f"{job_id}.log"),
                        result_path=str(result_path) if result_path else None, result_kind=kind)
    with _lock:
        _queue.append(job_id)
    return _public(JOBS[job_id])


def _public(j):
    return {k: v for k, v in j.items() if k not in ("log_path", "result_path")}


@app.get("/jobs")
def list_jobs():
    return [_public(j) for j in sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return _public(JOBS[job_id])


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    p = Path(JOBS[job_id]["log_path"])
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    j = JOBS[job_id]
    if not j["result_path"] or not Path(j["result_path"]).exists():
        raise HTTPException(404, "no result yet")
    if j["result_kind"] == "json":
        return json.loads(Path(j["result_path"]).read_text(encoding="utf-8"))
    return FileResponse(j["result_path"], media_type="text/markdown")


@app.get("/health")
def health():
    from . import __version__
    return {"ok": True, "version": __version__, "queued": len(_queue), "work": str(WORK)}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
