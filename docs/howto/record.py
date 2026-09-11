"""Record the how-to video: the page driven end to end, captioned, against a real dataset.

    python docs/howto/record.py --dataset E:/work/Isaacsim/so101-scene/datasets/can_v3d --out docs/howto

Starts `omnibase serve` on a spare port, drives it with Playwright while recording, then
writes howto.mp4 and howto.gif with ffmpeg. Captions are injected into the page for the
recording only; the product has none.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--out", default="docs/howto")
ap.add_argument("--port", type=int, default=8791); ap.add_argument("--episodes", default="0,1,2,3,4,5,6,7")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
work = out / ".work"; shutil.rmtree(work, ignore_errors=True); work.mkdir()
url = f"http://127.0.0.1:{a.port}/"

srv = subprocess.Popen([sys.executable, "-m", "omnibase", "serve", "--port", str(a.port)], env={**os.environ, "OMNIBASE_WORK": str(work.resolve())},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(work))
for _ in range(60):
    try:
        import urllib.request; urllib.request.urlopen(url + "health", timeout=1); break
    except Exception:
        time.sleep(0.5)

CAPTION = """
(() => { let el = document.getElementById('__cap'); if (!el) { el = document.createElement('div'); el.id = '__cap';
  el.style.cssText = 'position:fixed;left:50%%;bottom:22px;transform:translateX(-50%%);max-width:70ch;padding:9px 16px;background:oklch(0.21 0.012 60);color:oklch(0.94 0.01 80);font:15px/1.4 ui-monospace,Menlo,Consolas,monospace;z-index:9;box-shadow:0 2px 12px rgba(0,0,0,.25);transition:opacity 200ms';
  document.body.appendChild(el); } el.textContent = %s; el.style.opacity = %s ? 1 : 0; })()
"""

def cap(page, text, hold=2.2):
    page.evaluate(CAPTION % (json.dumps(text), 'true' if text else 'false')); page.wait_for_timeout(int(hold * 1000))

def wait_done(page, timeout=180):
    page.wait_for_function("document.getElementById('st') && /done|failed/.test(document.getElementById('st').textContent)", timeout=timeout * 1000)

def type_into(page, sel, text):
    page.click(sel); page.fill(sel, ""); page.type(sel, text, delay=28)

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=str(work / "video"), record_video_size={"width": 1280, "height": 800}, color_scheme="light")
    page = ctx.new_page(); page.goto(url); page.evaluate("localStorage.clear()"); page.goto(url); page.wait_for_timeout(800)
    cap(page, "omnibase: where would the robot have to stand? Four steps, one plan.", 3.2)
    cap(page, "1  place: give it a LeRobot dataset of hand-held demonstrations.", 1.2)
    type_into(page, "#f_dataset", a.dataset); page.wait_for_timeout(500)
    type_into(page, "#f_hands", "right"); type_into(page, "#f_home", "0.18,0.28,0.08")
    cap(page, "--home is where the robot really stands, if you already know. Everything else has a sensible default.", 2.4)
    page.click("summary"); page.wait_for_timeout(900)
    type_into(page, "#f_tcp", "0.0748"); type_into(page, "#f_stride", "3"); type_into(page, "#f_workers", "8"); page.click("summary"); page.wait_for_timeout(400)
    page.evaluate("document.getElementById('f_dataset').value = document.getElementById('f_dataset').value")   # keep as typed
    # a shorter sweep for the recording: eight episodes
    page.evaluate(f"(() => {{ const f = document.getElementById('form'); const i = document.createElement('input'); i.name = 'episodes'; i.value = '{a.episodes}'; i.type = 'hidden'; f.appendChild(i); }})()")
    cap(page, "The line under the form is the command that runs. The page is the command line with a face.", 2.6)
    page.click("#run")
    cap(page, "It sweeps every candidate base for every episode: which frames could a fixed robot execute from here?", 3)
    wait_done(page)
    cap(page, "The map: darker = more frames executable from that base. Amber = where to stand. Red = where the robot stands now.", 3.4)
    page.evaluate("document.querySelector('canvas').scrollIntoView({block:'center'})"); page.wait_for_timeout(2200)
    page.evaluate("window.scrollTo(0, 0)"); page.wait_for_timeout(600)
    cap(page, "The sweep wrote a plan: one base per window of every episode. The next steps read it.", 2.4)
    page.click("text=use this plan → report"); page.wait_for_timeout(1200)
    cap(page, "2  report: the dataset as that robot sees it.", 1.4)
    page.click("#run"); wait_done(page)
    cap(page, "What a fixed base loses against one base per window; grasp events; still frames; the workspace.", 3.2)
    page.evaluate("document.querySelector('pre.md') && document.querySelector('pre.md').scrollIntoView({block:'start'})"); page.wait_for_timeout(2600)
    page.evaluate("window.scrollTo(0, 0)"); page.wait_for_timeout(400)
    page.click("text=3 ambiguity"); page.wait_for_timeout(900)
    cap(page, "3  ambiguity: before you augment across bases, what would averaging the joint actions cost?", 2.4)
    type_into(page, "#f_spans", "2,4"); page.click("summary"); page.wait_for_timeout(500); type_into(page, "#f_limit", "12"); page.click("summary")
    page.click("#run"); cap(page, "Nothing is trained. The demonstrations are re-solved from a grid of bases and the actions compared at the tool.", 3); wait_done(page)
    cap(page, "Absolute joint targets miss by about the spread of the bases; joint deltas by under a centimetre. Export with --action delta.", 3.6)
    page.wait_for_timeout(800)
    page.click("text=4 probe"); page.wait_for_timeout(900)
    cap(page, "4  probe: does a policy know where it stands, or did it memorise the motion?", 2.4)
    type_into(page, "#f_offsets", "2,4,8"); page.click("summary"); page.wait_for_timeout(500); type_into(page, "#f_knn_grid", "-4,0,4"); type_into(page, "#f_limit", "12"); page.click("summary")
    page.click("#run"); cap(page, "Move the base, re-solve the state, ask the policy, measure the miss at the tool. No rollout.", 3); wait_done(page)
    cap(page, "Replay misses by exactly the shift; the oracle by nothing. A policy fit at one base sits on the replay line; one fit on a grid stays flat.", 4)
    page.evaluate("document.querySelector('table') && document.querySelector('table').scrollIntoView({block:'center'})"); page.wait_for_timeout(2000)
    cap(page, "Plug in your own policy with --policy module:attr. None of this predicts success — that is a rollout's job.", 3.4)
    cap(page, "pip install 'omnibase[data,service]'   ·   omnibase serve", 3)
    cap(page, "", 0.4)
    ctx.close(); browser.close()

srv.terminate()
webm = next((work / "video").glob("*.webm"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "howto.mp4")], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "howto.mp4"), "-vf", "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4", str(out / "howto.gif")], check=True)
shutil.rmtree(work, ignore_errors=True)
print("wrote", out / "howto.mp4", out / "howto.gif")
