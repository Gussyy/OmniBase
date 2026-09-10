# Experiment log: FastUMI → OmniBase → SO-101 → SmolVLA → tomato-soup-can pick, zero-shot

**Date:** 2026-09-09 (one day, continuing overnight). **Operator:** Claude (pen name Jeff on the
shared board), with the user's decisions where noted. **Status at writing (23:50):** the
corrected re-sweep is running; the corrected export, training and rollouts are queued behind it.
Everything marked *pending* is filled in when it lands. Nothing here is projected.

This is the working record: what was decided, what was measured, what went wrong, what was
learned, and what runs next. The polished version is `docs/FASTUMI_POC.md`.

---

## 1. Goal and the decisions that shaped it

**Claim under test.** OmniBase turns hand-held gripper recordings (no robot in them) into
SO-101 joint-space training data by choosing a base placement per policy window. Prove it is
useful: select pick-and-place data from FastUMI-100K, retarget it, train a vision-language-action
policy that has never seen robot data, and pick a tomato-soup can into a box zero-shot with the
robot fixed in place.

| decision | choice | who |
|---|---|---|
| source data | `IPEC-COMMUNITY/FastUMI_100k_lerobot` only (RealOmni is a gated 95 TB mcap set) | user |
| extra data allowed | the 16 VR gripper episodes `so101-scene/datasets/can_v3` (HF `Rachata/Umi-Sim-Gripper_pick-tomato-can`), no more | user |
| deployment | Isaac Sim (`so101-scene`) only | user |
| camera | wrist fisheye only, as UMI has | user |
| policy | LeRobot SmolVLA architecture, VLM initialised from `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, action expert from scratch | user |
| target object | the **tomato soup can** the VR episodes were recorded with, not the LeHome tomato prop | user (clarified 22:30) |
| task string | one string for every source: "pick up the object and put it in the container" | Claude |
| budgets | ≤100 GB downloads, ≤$10 RunPod as fallback, local RTX 4070 Ti 12 GB | user |
| training length | 5 h per policy (32,500 steps at the measured 0.55 s/step) | user (23:48) |
| code policy | no large changes to OmniBase; use it as an interface; use the full GPU and CPU | user |

Spent so far: 31 GB of the download budget, $0 of the RunPod budget.

## 2. The pipeline as built

```
FastUMI parquet (30 kB/episode)  → omnibase level   (per-task frame calibration, no video)
                                 → omnibase sweep   (base per 21-frame window; chunked vs fixed yield)
                                 → omnibase select  (episodes worth their bytes, --min-held 0.7)
                                 → fetch_fastumi.py videos (only the chosen mp4s)
                                 → omnibase export  (IK per chunk, ffmpeg slice, LeRobot v3.0)
                                 → lerobot-train smolvla (batch 64, bf16, 12 loader procs)
                                 → lerobot_server.py policy ⇄ ZMQ ⇄ eval_policy.py (Isaac Sim)
```

Everything after `export` drives OmniBase through its CLI. Library additions made for this
work: `level`, `select`, `export`, `solve()` (joint angles kept, continuity across frames),
`write.py` (the v3.0 writer), `SO101_BODY_IN_CHAIN` (see §6). Simulator-side additions:
`eval_policy.py`, the policy server's `--degrees --absolute --match --rename --samples --dump`,
`simbridge/lerobot.py` conversions, `configs/eval_tomato_box.yaml`, three measurement scripts.

## 3. Data: what FastUMI is and what had to be measured

FastUMI writes `x, y, z, roll, pitch, yaw, gripper` per frame, absolute in the tracker's pose at
frame 0 (RealSense T265: x right, y up, z back), gripper 0.04 closed to 1.0 open, 720x1280
fisheye at 20 fps, ~10 MB video + 30 kB parquet per episode. It does not say which way is up,
where the tracker sits on the gripper, which Euler convention it used, or which way the jaws
point in their own frame.

**Per-task calibration (`omnibase level`), all still valid:**

| task | euler | tool offset | contacts flat to | under the table | fingers vs table | `aligned` |
|---|---|---|---|---|---|---|
| put_shoes_into_storage_box | xyz | 0.15 m | 22 mm | 2.7% | 14° | 0.95 |
| Prepare_tableware | xyz | 0.00 m | 18 mm | 1.8% | 16° | 0.93 |
| get_plate_and_spoon_from_dish_rack | ZYX | 0.03 m | 23 mm | 2.5% | 19° | 0.92 |
| make_sandwich | ZYX | 0.00 m | 15 mm | 0.1% | 22° | 0.89 |
| take_items_out_of_drawer | ZYX | 0.06 m | 10 mm | 3.2% | 6° | 0.99 |

How each is measured: **up** from contact-point flatness (positions only, levelled per
episode because a rig docked in the same slot shares orientation but not origin); **the sign
of up** by whichever leaves fewer frames underground; **the finger axis** as the body direction
pointing at the table when the jaws close; **the Euler order** by that consistency (`aligned`);
**the tool offset** by which offset collapses grasps at different wrist angles onto one surface
(weakly identified, 5 cm scatter over an effect worth 1; the yield was checked to be insensitive).

The calibration lives in the gripper *body's* frame (fingers +y, housing top +z) and was not
affected by the frame bug of §6.

## 4. Sweep, selection, export, training: the first pass (wrong frame, numbers to be replaced)

These numbers were computed with the tool point on the servo and the jaw axis 90° off (§6).
They are kept because they are what the day produced; the corrected re-sweep replaces them.

| task | episodes swept | frames | chunked | one fixed base | eps ≥90% held |
|---|---|---|---|---|---|
| Prepare_tableware | 1500 | 225,545 | 89.3% | 54.9% | 728 |
| get_plate_and_spoon_from_dish_rack | 600 | 89,291 | 81.5% | 40.9% | 37 |
| make_sandwich | 600 | 88,731 | 81.7% | 43.4% | 254 |
| put_shoes_into_storage_box | 600 | 114,273 | 75.5% | 33.3% | 2 |
| take_items_out_of_drawer | 600 | 106,752 | 90.7% | 58.6% | 340 |
| all | 3900 | 624,592 | 84.8% | 48.0% | 1361 |
| can_v3 (16 VR episodes) | 16 | — | 95.0% | 81.7% | — |

Reach was never the limit (100% of frames within 35 cm of some base); the 5-DoF wrist was.
That statement survives the fix; the percentages do not. First corrected number: **can_v3
77.1% chunked vs 73.1% fixed** with the real jaws.

**Selection:** `--min-held 0.7 --budget-gb 90` → 3,307 episodes, 32.2 GB, 445k usable frames
(1418 tableware, 597 drawer, 524 plate, 413 sandwich, 355 shoes). The yield threshold is the
real selector; bytes barely discriminate because every episode is ~10 MB.

**Export (first pass):** OmniBase set 8,463 episodes / 256,433 frames / 1.4 GB in 17 min; fixed
twin 4,427 episodes / 155,945 frames / 815 MB in 9 min (12 processes; Ken's solver rewrite is
why it is minutes, not hours). Per-joint step per frame median 1.5°, p95 9.3°; the wrist roll
reverses direction in 0.7% of frames, so no jitter. Both verified: FK of exported angles within
14.8 mm / 20.0° of the hand (in the same wrong frame), action = next state, all clips match
their row counts, stats correct.

**Training (first pass):** 25,000 steps, 3 h 52 min, loss 2.10 → 0.41 (1k) → 0.21 (5k) →
0.153 (10k) → 0.110 (25k), no plateau, no spikes. Curve: `loss_run1_wrong_frame.png`.

**Throughput measurements (still valid):**

| setting | samples/s | note |
|---|---|---|
| fp32, batch 32, 12 workers | 119 | loader wait 3 ms/step: the GPU is the limit |
| bf16, batch 64 | 112 | 8.8 GB peak, chosen |
| bf16, batch 96 | 97 | 11.0 GB peak, 1 GB headroom |
| any batch 16–96 | ~100–120 | flat: the 450M model saturates the card |

LeRobot's `--policy.use_amp` crashes on this stack (fp16 GradScaler cannot unscale the bf16
VLM's gradients); `ACCELERATE_MIXED_PRECISION=bf16` with `use_amp=false` works.

## 5. Rollouts and the diagnosis

Three evaluations of the first policy on the can/tomato scene, 20 episodes each:

| variant | reached | lifted | placed |
|---|---|---|---|
| as trained | 0/20 | 3/20 (knocked) | 0/20 |
| corrected camera mask | 0/18 (server died) | 0 | 0 |
| corrected mask + 8 averaged draws | 0/20 | 0/20 | 0/20 |

**Offline diagnostics on CPU (`E:\data\out\offline_*.py`), on the policy's own training frames:**

- Single-step prediction error ~6° per joint, worse than holding still (2.2°). Expected from a
  flow-matching loss of 0.11 in normalised units (≈4–6°).
- Whole 10-step chunk on FastUMI frames: error 19.1° at horizon 10 vs 25.4° holding still,
  **85% direction agreement**. The policy learned the motion direction on real data.
- On the slow VR-can frames: 11.4° vs 7.2° holding still, 69% agreement: noise dominates.
- Draw-to-draw spread 7.8–8.2°: the error is mostly **sampling noise**. The mean of 6 draws
  halves it (12.1° → 6.5° on FastUMI). Hence `--samples N` on the server.

**Camera mask:** the export recorded the union of lit regions, which the VR-can frames (full
rectangle) and the fisheye vignette stretched to the full frame; the real picture is 0.37 ×
0.374 of the frame (median over 120 frames at 40/255). Both datasets' `meta/omnibase.json`
were corrected.

**What the policy saw in the simulator** (`--dump`): a white void, a speckled black table at
the bottom edge, no object; the arm folded over its own base. Camera moved to the VR
recording's pose (0, −0.10, 0.14) in the gripper frame, 31° down, DLAA + denoiser. After that
the frames looked like the recording: grey table, box, can, fingers at the bottom.

None of that changed the score, which is what led to measuring the arm itself.

## 6. The three bugs, found by writing joint angles into the simulator and reading link positions

**Method.** `so101-scene/scripts/joint_convention_check.py` writes joint angles kinematically
(no gravity, no controller), steps once, reads `gripper_base`, `gripper_frame_link`, `arm_r`,
`arm_l` positions relative to the root, and compares with OmniBase's chain for the same angles.
`playback_check.py` plays a retargeted VR-can grasp frame back and compares the arm's finger
direction with the recorded gripper's. `fk_check.py` commands a pose through the evaluator's
action path and measures what the arm actually holds.

**Bug 1 — OmniBase's gripper frame was 90° off (library).** The chain ends at the wrist_roll
joint's frame, not the gripper body's. In the terminal frame, for every pose tested (zero, each
single joint at ±45°, the ready pose): the fingertip frame sits at (−0.008, 0, −0.098), the
servo at (0, −0.082, 0), the jaw-opening axis along (−0.93, 0, 0.36). So fingers = −z, housing
top = −y, jaws open along x. `SO101_APPROACH` said +y and `SO101_TOOL` was (0, −0.0748, 0): the
tool point sat on the servo. Joint conventions themselves agree exactly between chain and
simulator (the finger direction is the same in the chain's frame under every joint move). The
+y had been "confirmed" against the VR recording, which reports the gripper *body's* frame,
where +y is right. Every exported trajectory aimed the real fingers 90° from the human's; the
sweep was just as feasible, and the export's FK check agreed to the millimetre, because both
lived in the same frame.

*Fix (commit `915b25e`):* `SO101_BODY_IN_CHAIN = [[-1,0,0],[0,0,-1],[0,-1,0]]` (the body's axes
in the terminal frame; its own inverse), `SO101_TOOL = (0,0,-0.0748)`, `SO101_APPROACH =
(0,0,-1)`. `data.load(terminal=True)` turns every hand by it on the way to the solver;
`_apply_frame` and `fit_level` stay in the body frame; the tool-point shift uses `FINGERS =
BODY @ APPROACH = +y`. The VR can recording needs `--tcp 0.0748` now, because its `pos` is the
body origin and the pads are 75 mm out.

*Validation:* replaying retargeted can grasps in the simulator puts the arm's fingers **3.6°,
8.8°, 13.6°** from the recorded gripper's (was ~90°). Ken's solver is a general damped
least-squares IK and is fine with the off-axis tool: round trip 0.26° median / 2.9° p90 (was
0.01° when the roll was a free joint); two test bounds moved accordingly.

**Bug 2 — the simulator arm could not go where it was told (bridge).** The task's arm action
is `default_joint_pos + 0.5 × action`; `simbridge.lerobot.degrees_to_array` computed the offset
against the *stock* task defaults and clipped to ±1, i.e. ±28.6° around a pose this scene does
not use. For the ready pose it sent (0, −1, +1, −0.08, 0): the shoulder settled 28.6° low and
the elbow on its stop, identical with stiffness 200 or 1000, which is how it was recognised as
not being gravity. *Fix (commit `ce5294f`):* server `--absolute` sends radians; `eval_policy.py
--absolute` finishes the conversion with the live environment's default pose and scale
(`absolute_to_env`). Tracking now 0.1° on held and commanded poses (1.0° on the wrist roll).

**Bug 3 — the camera aimed at nothing (scene).** Minor; fixed as above.

**Also fixed on the way:** actuator gains 1000/20/effort 50 (200/5/10 could not hold the arm
under the old action path; kept stiff because the training targets are exact angles); ready
pose solved with OmniBase's own chain (tool 18 cm ahead, 16 cm up, jaws 45° down); the base 8 cm
above the table because every sweep placed it there; object = YCB `tomato_soup_can` at
(0.26, −0.06, 0.06).

## 7. Guards so it does not happen again

- `OmniBase/tests/test_robots.py` (commit `9be213f`): the chain's tool point and approach
  axis against gripper_base and fingertip positions measured in the simulator for seven joint
  configurations, tool within 12 mm and jaws within 8°. The old frame fails it at 75 mm / 90°,
  and that failure is itself a test. Runs in a second without Isaac Sim.
- `eval_policy.py` preflight (commit `7e71a20`): holds the reset pose through the policy's own
  action path for 30 steps and refuses to score if the arm drifts more than 2°.
- README rule: measure a robot's frame in the simulator against link positions; never against
  a recording, never against the model's own forward kinematics; paste the table, keep the test.
- The three measurement scripts are committed; re-measuring an arm is minutes.

## 8. Operational learnings (Windows, LeRobot 0.6.1, two agents on one box)

- LeRobot refuses an existing `--output_dir`; do not pre-create it for a log.
- Checkpoint symlinks need the `mklink /J` junction fallback in `train_utils.py`.
- torchcodec's DLL fails to load; `--dataset.video_backend=pyav`. Its tracebacks at startup
  are noise.
- The step field in logs is rounded past 1K; count log lines for the loss curve.
- A policy server started with PowerShell `Start-Process` gets a visible console; one was
  closed and the server died with `forrtl: window-CLOSE`. Use `-WindowStyle Hidden`. Capturing
  its PID through `$(...)` hangs because the detached child keeps the pipe open; write the PID
  to a file. Kill it by PID and then by command line, because one survived and blocked the port.
- Never filter processes by a string that appears in your own command line; the kill takes
  out your own shell. Exclude `*shell-snapshots*`.
- A pause script must know every pipeline name; one that missed `pipeline4.sh` let it jump
  ahead and start a second server on the same port.
- The verifier must average in float64; float32 over 256k rows drifts 0.02°.
- `omnibase sweep` writes its plan only at the end. A power cut at 00:40 on 2026-09-10 threw
  away 1.7 h of the 1,418-episode tableware task. The re-sweep now runs in parts of 200
  episodes (`plans2/parts/`, merged by `merge_parts.py`), every pipeline stage skips itself once
  its result exists, training resumes from its last checkpoint, and one command restores the
  whole chain after a reboot: `bash E:\data\out\resume_all.sh`.
- A guard that has never run is not a guard. The evaluator preflight (added 23:47) first ran at
  14:50 the next day and crashed on an unset variable; fixed, it then *passed* while parking the
  arm straight up: it read the packet's radians as degrees, converted again, and compared a
  1.43-radian drift against a 2-degree threshold. The 0/20 it let through is kept as
  `eval_omnibase2_badpreflight.json`. Rule now: units in variable names (`q0_rad`, `start_deg`),
  and every guard is exercised once on a known-good and once on a known-bad input before it
  is trusted. The preflight also checks that the reset pose is the scene's default (≤15°).
- The corrected tool point costs solver time: with the jaws off the roll axis `chain.ik` runs
  3.7× slower per solve (can_v3, identical 0.36 M solves: 251 s against 68 s), so the re-sweep
  and the exports take 3.7× the first pass. Same answers, more seeds; posted to Ken.
- Everything coordinated with Ken (the other agent) through `session.md`: claims per file,
  the feasibility-mask gate (protects the solver; the robot model change legitimately moves
  its reference), CPU/GPU windows, and a running log with real clock times.

## 9. What runs next (all queued, automatic)

1. Re-sweep of the 3,307 selected episodes + can with the corrected frame:
   `E:\data\fastumi\resweep.sh` → `plans2/`, in 200-episode parts, restarted 00:48 after the
   power cut, ~6.5 h (the corrected frame solves 3.7× slower than the first pass).
2. `E:\data\out\pipeline6.sh`: export `so101_omnibase2` and `so101_fixed2` from `plans2`,
   verify both, train `so101_omnibase2` 32,500 steps (~5 h), 20 rollouts with `--absolute`
   and 8 averaged draws, then the fixed twin at the same steps and its 20 rollouts.
3. Regenerate `docs/FASTUMI_POC.md` tables from `plans2` and the new datasets; write the
   tomato-can result, cost, and "what this does not show".

Rough ETA (revised 00:55 after the power cut): corrected yields ~07:30, exports ~09:00, first
rollout numbers ~14:00, twin's ~19:30.

If the rollouts still score 0, the levers inside the pipeline, in order: fine-tune the finished
policy 3,000 steps on the 16 can episodes (`finetune.sh`, allowed by the rules), continue
training, more averaged draws. No new data.

## 10. Results (pending)

| policy | steps | final loss | reached | lifted | placed |
|---|---|---|---|---|---|
| so101_omnibase2 | 32,500 | pending | pending | pending | pending |
| so101_fixed2 | 32,500 | pending | pending | pending | pending |

### Corrected sweep (right frame, the 3,307 selected episodes + can), finished 08:42 on 2026-09-10

Frame-weighted, window 21 at stride 2, base grid 0.06 m over 0.42 m, height 0.08 m; the can at
stride 3, step 0.05, `--tcp 0.0748`. Generated by `E:\data\out\sweep_table.py plans2`.

| task | episodes swept | frames | chunked | one fixed base | episodes ≥90% held |
|---|---|---|---|---|---|
| `Prepare_tableware` | 1418 | 213,289 | **96.2%** | 75.4% | 1244 |
| `get_plate_and_spoon_from_dish_rack` | 524 | 78,348 | **93.1%** | 63.5% | 416 |
| `make_sandwich` | 413 | 62,042 | **53.0%** | 24.4% | 5 |
| `put_shoes_into_storage_box` | 355 | 68,877 | **83.8%** | 41.0% | 78 |
| `take_items_out_of_drawer` | 597 | 106,226 | **88.3%** | 49.5% | 277 |
| `can_v3` (16 VR episodes) | 16 | 1,230 | **77.1%** | 73.1% | 2 |
| **all** | 3323 | 530,012 | **87.4%** | 58.0% | 2022 |

What the frame fix did, on the same episodes (first pass = wrong frame, tool on the servo, jaws
90° off):

| task | wrong frame chunked / fixed | right frame chunked / fixed |
|---|---|---|
| `Prepare_tableware` | 90.3 / 55.7 | 96.2 / 75.4 |
| `get_plate_and_spoon_from_dish_rack` | 82.7 / 41.7 | 93.1 / 63.5 |
| `make_sandwich` | 90.1 / 48.7 | **53.0 / 24.4** |
| `put_shoes_into_storage_box` | 78.8 / 34.6 | 83.8 / 41.0 |
| `take_items_out_of_drawer` | 90.8 / 58.7 | 88.3 / 49.5 |
| `can_v3` | 95.0 / 81.7 | 77.1 / 73.1 |

Reading it: the right frame is not uniformly harder. Four tasks got easier (the jaws sit 75 mm
further out than the servo did, so the wrist stays further from the table and the base), the
sandwich and the can got much harder (their grasps come in low and flat, and aiming real jaws
at them costs reach). The single-fixed-base column rose too, so OmniBase's margin over one base
narrows from roughly 36 to 29 points overall, and on the sandwich the absolute yield halves. The
selection (`selection5.json`, ≥70% held) was made on the wrong-frame numbers; the export keeps
every contiguous run of ≥20 frames, so low-yield episodes still contribute their reachable
stretches. Cost: 2.1× the solver time of the first pass (8.1k solves/s against 17.3k).

## 10b. The clean 0/20, and what it was made of (2026-09-10, 15:30-18:30)

With the corrected frame, the corrected action path and a valid preflight, `so101_omnibase2`
still scored 0/20: every episode timed out with the arm panning away from the can (+43° in 40
frames) and the jaws never closing. The dumped frames showed the can dead-centre between the
fingers at the ready pose. Four things were wrong at once, all in how *I* had prepared the can
demonstrations and the evaluation scene, none in the FastUMI side:

1. **The can was swept without `--home`.** Every one of the 16 VR episodes had its base placed
   on the grid 3-28 cm from the eval robot's actual base; `at_home` was 0 for all of them. Joint
   space is base-relative, so the can data described an arm standing somewhere else, and the
   policy panned toward where that arm's joints would be. OmniBase has the flag for exactly this;
   I had not passed it.
2. **With `--home 0,0,0.08`, 0.0% of the can frames are feasible at 20°.** The operator's
   approach axis is yawed 40-118° from the base bearing (pitched down 54-85°): a 5-DOF arm's
   tool must lie in the vertical plane through its base, and a human hand does not. Loosening
   the orientation tolerance -- physically free for an upright cylinder -- gives 31% at 45°, 38%
   at 60°, 60% at 75°, 72% at 90° (position within 15 mm throughout).
3. **The eval can stood 8 cm from where every demo grasped it.** The recordings grasp at
   x 0.186-0.207, y -0.038-0.000 (root frame; the collector writes poses in the robot's root
   frame); the eval had (0.26, -0.06), read last night off the wrong-frame grasp estimate. Moved
   to (0.20, -0.01) with the demos' own ±2 cm of spread.
4. **The VR demos are three-quarters pause.** The operator moves in bursts; resting rows repeat
   to the millimetre. 55-75% of the exported can rows had a zero joint step (FastUMI: 8%), so the
   policy was taught to hold still on the one scene it is tested in. Not a slow sampler -- the
   moving stretches are smooth at 30 fps -- so no interpolation: `E:\data\out\drop_holds.py`
   writes `datasets/can_v3d`, the same 16 episodes with the paused rows and their video frames
   removed (3,674 → 1,934 rows, 53%). Motion per 10 Hz row: 7 mm median where it was 0.

Also seen: the VR renders are full-frame (no fisheye border), so the 0.37 ellipse the eval masks
to -- right for the FastUMI frames -- is not what the can frames looked like in training; the
fine-tuned policies are evaluated with the can set's own (full-frame) mask. And the eval robot's
fingers are yellow where the VR gripper's were grey; left as is (the user's call: visuals last).

What runs on it: `can_v3d` swept at home with rot-tol 90 (43% of moving rows at home, episodes
0/6/11 included), filtered to the at-home chunks (`home_only.py`), exported ×8 as `can_home2`
(1,952 rows, 80 episodes, median step 2.1°), both policies fine-tuned 3,000 steps on it on the
rented 4090, then 20 rollouts each in the corrected scene -- alongside the two plain policies
in the corrected scene (0.37 mask) for the like-for-like pair.

## 10c. The robot where the demos are executable, and a fourth frame bug (21:10)

`omnibase sweep --home` over the pause-free demos says the fixed SO-101 executes 45% of their
moving rows from (0.18, 0.28) at the strict 15 mm / 20° -- the operator worked from the left --
and none from (0, 0). So the last experiment fixes the robot there (`eval_tomato_box_base2.yaml`:
can and box unmoved, ready pose = a demo's own start posture, place target written in the root
frame because `to_root_frame` rotates but does not translate). `can_base2` = the seven at-home
chunks of the six long grasp episodes, ×8; offline, its joints put the jaws within 5.9 mm /
6.6° of the recorded hand down to the can.

Fine-tuned on it, the OmniBase policy **descends to the can, closes the jaws 22 mm from its
centre, and lifts** -- the first grasp attempt of the whole run. The jaws closed beside the can
(a nearly blind policy replaying the mean demo against ±2 cm of can jitter; the moving jaw
pushes the can), so `lifted` stayed 0. `reached` also stayed 0, which is the fourth bug: the
evaluator's `SO101_FULL_GRASP_OFFSET` was (0, -0.0748, 0) in gripper_base -- the SERVO
direction, the same 90° mistake the retargeting had, so "reached" was scored 10 cm from the
jaws and no correct grasp could ever satisfy it (day one's "lifted without reached" was this).
Fixed to (0, 0, -0.0748) from the measured fingertip vector, guarded by
`so101-scene/tests/test_tuning.py`, and the base2 rollouts were rerun. Lesson, again: every
constant that names a direction on the robot -- tool, approach, camera, *and metric* -- gets
checked against a measurement, not against another constant.

## 10d. The can is picked up (2026-09-10, 22:00-23:20)

Three more things were wrong, all of them in the scene and the measuring, none in the data:

4. **The evaluator's grasp point pointed at the servo.** `SO101_FULL_GRASP_OFFSET` was
   (0, -0.0748, 0) in `gripper_base` -- the same 90 degree error the retargeting had, surviving
   in the metric. "reached" was scored 10 cm from the jaws, so no correct grasp could satisfy it.
   Fixed from the measured fingertip vector; `so101-scene/tests/test_tuning.py` guards it. The
   base2 policy went from 0/20 to **20/20 reached** on the same rollouts.
5. **The can was lying down and the jaws closed through it.** The YCB can's mesh is 68 x 102 x 68
   mm with its long axis along local Y, and the scene passed no rotation, so it lay on its side.
   `collider_probe.py` measured that, and also that this path reads the config quaternion as
   (x, y, z, w) -- the default (0,0,0,1) is identity, which is exactly the lying box observed.
   +90 degrees about x stands it up. Separately, the jaws closed at full travel: 30 mm through a
   68 mm can, and straight through each other when empty (there is no finger-to-finger collision),
   which is what the rollout videos showed. They now stop 6 mm inside the can's surface.
   Before: a replayed demonstration never moved the can by a millimetre. After: it moves.
6. **Episodes after the first were not fresh trials.** `--max-steps` is the evaluator's own
   counter, not the task's, so Isaac Lab never reset those environments and each episode carried
   on from where the last stopped -- measured pixel-identical across the boundary, and nearly
   every episode ended on the cap. Found because the user noticed episode 2's first video frame
   was episode 1's last. The scene is now reset explicitly when the cap ends an episode.

With those three fixed, the OmniBase policy **picks the can up in 4 of 4 episodes** at the
demonstrated position, from a fresh reset each time, and the fixed-base twin reaches equally well
(4/4) and lifts **0 of 4**. `placed` stays 0: the at-home chunks the fine-tune set is built from
cover the approach and the grasp and mostly stop before the transport to the box.

The lesson that keeps repeating, now four times over: every constant that names a direction or a
frame -- tool, approach, camera, metric -- must be checked against a measurement, never against
another constant. And a guard or a counter that has never been exercised is not one.

## 11. Where everything is

| what | where |
|---|---|
| plans (wrong frame / corrected) | `E:\data\fastumi\plans\` / `E:\data\fastumi\plans2\` |
| level fits | `E:\data\fastumi\level\` |
| selection | `E:\data\fastumi\selection5.json` |
| videos | `E:\data\fastumi\single_arm\<task>\videos\` (31 GB) |
| datasets (first pass / corrected) | `E:\data\out\so101_omnibase`, `so101_fixed` / `so101_omnibase2`, `so101_fixed2` |
| checkpoints | `E:\data\out\train_<dataset>\checkpoints\` |
| rollout results | `E:\data\out\eval_<label>.json`, frame dumps `dump_*` |
| launchers | `E:\data\out\{resume_all,run6,pipeline6,train,eval,finetune,verify,plot_loss,sweep_table}.*`, `E:\data\fastumi\{resweep.sh,merge_parts.py}` |
| logs | `E:\data\fastumi\resweep.log`, `E:\data\out\pipeline6.log`, `train_*.log`, `server_*.log` |
| offline diagnostics | `E:\data\out\offline_{check,chunk,noise,avg}.py`, `measure_circle.py` |
| measurement scripts | `so101-scene/scripts/{joint_convention_check,playback_check,fk_check}.py` |
| shared board | `OmniBase/session.md` (gitignored) |
