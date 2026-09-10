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
- [ ] **Export** `so101_omnibase2` and `so101_fixed2` from `plans2` (can with `--tcp 0.0748`, ×8 repeats)
- [ ] **Verifier passes** on both (FK ≤15 mm/20°, rows, clips, stats)
- [ ] Check `meta/omnibase.json` ellipse ≈ 0.37 × 0.374 on both *(manual)*
- [ ] **Train `so101_omnibase2`** 32,500 steps (~5 h, ~09:45→14:45); checkpoints every 2,500; loss curve plotted with `plot_loss.py OUT.png 2` *(manual: plot)*
- [ ] **Preflight passes** at rollout (arm holds reset pose ≤2°)
- [ ] **20 rollouts `so101_omnibase2`** (absolute angles, 8 averaged draws, frames dumped) → `eval_omnibase2.json`
- [ ] Look at the dumped frames: object in view, fingers where expected *(manual)*
- [ ] If 0/20: re-export `can_only2` from `plans2/can_v3.json`, fine-tune 3,000 steps, roll out again *(manual decision)*
- [ ] **Train `so101_fixed2`** 32,500 steps; **20 rollouts** → `eval_fixed2.json`
- [ ] Regenerate `docs/FASTUMI_POC.md`: sweep table from `plans2`, dataset table, training, tomato-can results, cost, "what this does not show" *(manual)*
- [ ] Fill §10 of the experiment log; commit; post final numbers on the board and to the user *(manual)*
- [ ] Update project memory with the outcome *(manual)*
- [ ] Ken regenerates `F_reference.npy` for the new constants (their call; noted on the board)
- [ ] Decide what to keep of the wrong-frame artefacts (`so101_omnibase`, `so101_fixed`, `train_*`, ~5 GB): keep for the record unless disk is needed *(manual)*
