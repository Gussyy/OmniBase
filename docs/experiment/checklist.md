# Checklist — FastUMI → SO-101 → SmolVLA → tomato-soup-can pick

Ticked as things land. Times are local. Companion to `2026-09-09_fastumi_so101_smolvla.md`.

## Done (2026-09-09)

- [x] Calibrate 5 FastUMI tasks (`omnibase level`), `aligned` 0.89–0.99
- [x] Sweep 3,900 episodes, select 3,307, download 31 GB
- [x] Export both datasets, verify, train SmolVLA 25k steps (loss 0.110), roll out: 0/20
- [x] Diagnose: policy learns direction on real frames (85%), error is sampling noise (8° spread)
- [x] Find and fix bug 1: OmniBase gripper frame 90° off (`915b25e`), validated in-sim 3.6–13.6°
- [x] Find and fix bug 2: bridge action offset/clip (`ce5294f`), tracking 0.1°
- [x] Fix camera pose, renderer, object (soup can), gains, ready pose
- [x] Guards: `tests/test_robots.py`, evaluator preflight, README rule
- [x] Experiment log written and committed
- [x] Re-sweep launched 23:11 (`resweep.sh` → `plans2/`), pipeline6 armed behind it (32.5k steps each)

## Next (automatic unless marked *manual*)

- [x] Power cut 00:40 killed the sweep mid-task (it writes only at the end): relaunched 00:48 in 200-episode parts, pipeline6 made re-runnable (skips done stages, resumes training), `E:\data\out\resume_all.sh` restores everything after a reboot (00:55)
- [x] Windows Update rebooted the box 04:38 (event 1074) and killed the sweep at 198/200 of drawer part 1; nothing auto-resumes on logon (by design, needs the user's OK) — relaunched 08:10 with `resume_all.sh`, 17 parts skipped (08:12)
- [x] **Re-sweep finished 08:42** (after two restarts); corrected yields recorded in the log §10: all 87.4% chunked vs 58.0% fixed, sandwich halved, can 77/73 (08:50)
- [x] ~~**Re-sweep finishes** (`RESWEEP DONE` in `E:\data\fastumi\resweep.log`, ~08:45 — the corrected frame solves 2.1× slower) — record per-task chunked vs fixed yields in the log §10 with `E:\data\out\sweep_table.py` *(manual)*~~
- [x] Writer records the camera picture as a median, not a union, before the export runs (23:58)
- [x] **Export** `so101_omnibase2` (349,382 frames / 9,313 eps, 18 min, 1.9 GB) and `so101_fixed2` (249,902 / 5,453, 11 min, 1.4 GB) from `plans2` (09:19)
- [x] **Verifier passes** on both: per-frame step median 1.4° p95 10.3/10.4°, episodes ≥20 frames (09:24)
- [x] `meta/omnibase.json` ellipse 0.373 × 0.374 (omnibase2), 0.374 × 0.374 (fixed2), centred; the median fix works (09:27)
- [x] **Train `so101_omnibase2`** 32,500 steps, done 14:50, final loss 0.116 (curve: `loss_run2_omnibase2.png`); local 09:24→12:16 to step 17,500, a RunPod RTX 4090 was tried at 12:10 (user's call, to finish faster): checkpoint + dataset uploaded, resumed from 17,500, but that host runs at 34 TFLOPS bf16 (healthy 4090 ~165) with a CPU 6× slower than this desktop → 63 samples/s against 102 locally; abandoned 12:42, local run kept (never stopped), ~$0.25 spent; checkpoints every 2,500; loss curve plotted with `plot_loss.py OUT.png 2` *(manual: plot)*
- [ ] **Preflight passes** at rollout (arm holds reset pose ≤2°)
- [ ] **20 rollouts `so101_omnibase2`** (absolute angles, 8 averaged draws, frames dumped) → `eval_omnibase2.json`
- [ ] Look at the dumped frames: object in view, fingers where expected *(manual)*
- [ ] If 0/20: re-export `can_only2` from `plans2/can_v3.json`, fine-tune 3,000 steps, roll out again *(manual decision)*
- [ ] **Train `so101_fixed2`** 32,500 steps — on a second RunPod 4090 (`pod2`, 165 TFLOPS, $0.79/h) from 13:17, in parallel with the local run; `pod_watch.sh so101_fixed2 pod2` brings the checkpoint home; then **20 rollouts** → `eval_fixed2.json`
- [ ] Regenerate `docs/FASTUMI_POC.md`: ~~sweep table from `plans2`, dataset table, training, cost~~ (done 09:40), tomato-can results, loss curves, "what this does not show" *(manual)*
- [ ] Fill §10 of the experiment log; commit; post final numbers on the board and to the user *(manual)*
- [ ] Update project memory with the outcome *(manual)*
- [ ] Ken regenerates `F_reference.npy` for the new constants (their call; noted on the board)
- [ ] Decide what to keep of the wrong-frame artefacts (`so101_omnibase`, `so101_fixed`, `train_*`, ~5 GB): keep for the record unless disk is needed *(manual)*
