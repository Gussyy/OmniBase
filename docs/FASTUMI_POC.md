# Can a robot learn from people who were not using one?

**A proof, run end to end: 100 GB of budget, human hands, and an arm that has never appeared in
its own training data.**

This document records what was done, what was measured, and what did not work. Numbers marked
`TBD` are still running; nothing here is projected.

---

## The question

OmniBase's claim is narrow and testable. A hand-held gripper recording never says where a robot
stood, because there was no robot — so the base pose is a free variable, and choosing it per
policy window rather than per episode turns "mostly unreachable" into "mostly usable".

Until now that claim rested on 16 episodes of a simulated two-handed can pick, recorded in VR
by the person who wrote the library. This runs it against **15,000 episodes of real human
manipulation** recorded by other people, for other robots, with no idea OmniBase existed:
[FastUMI-100K](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot), a GoPro and
a tracker strapped to a hand-held gripper, in LeRobot v2.1 format.

And then it spends the answer: the selected episodes are retargeted onto an
[SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100), a policy is trained on them, and the
policy is asked to put a tomato in a box — a robot it has never seen, an object it was never
shown, in a simulator no frame of its training data came from.

## What had to be true first

FastUMI writes seven numbers per frame: `x, y, z, roll, pitch, yaw, gripper`. It does not write
down what any of them mean.

- The poses are **relative to the tracker's own pose at frame 0**, so nothing says which way is
  up. Frame 0 of the video shows why that is recoverable: the gripper sits in a fixed
  3D-printed slot on the collection rig before every take, so the orientation is a property of
  the task even though the origin drifts between sessions.
- The tracker is not the fingertips. The offset is not published.
- `roll, pitch, yaw` is three numbers and twelve conventions, and the file does not say which.
- Which way the gripper points in its own frame is not stated either.

All four are wrong by default, and a wrong one does not fail — it reports that the data is
unreachable. `omnibase level` measures each from the part of the data that can see it, and the
statistic that validates the lot is **`aligned`**: how sharply one body direction points at the
fitted table at every grasp. It can only be high if the episodes share a frame, the angles were
read in the right order, and the fitted table is the table.

| task | euler | tool offset | contacts flat to | under the table | fingers vs the table | `aligned` |
|---|---|---|---|---|---|---|
| `put_shoes_into_storage_box` | xyz | 0.15 m | 22 mm | 2.7% | 14° | 0.95 |
| `Prepare_tableware` | xyz | 0.00 m | 18 mm | 1.8% | 16° | 0.93 |
| `get_plate_and_spoon_from_dish_rack` | ZYX | 0.03 m | 23 mm | 2.5% | 19° | 0.92 |
| `make_sandwich` | ZYX | 0.00 m | 15 mm | 0.1% | 22° | 0.89 |
| `take_items_out_of_drawer` | ZYX | 0.06 m | 10 mm | 3.2% | 6° | 0.99 |

The last two columns are the free checks. Neither is fitted against the table: the fingers come
from where they point at a grasp, the table from where contacts lie, and they agree to within
6–22° across five independently calibrated tasks.

**The tool offset is the one number that stays loose.** It is found by asking which offset makes
grasps at different wrist angles land on one surface, which works when objects are one height
and stops when they are not — a grasp happens at table height plus half an object, so assorted
objects write 5 cm of scatter over an effect worth 1. This is asserted as a test, not filed as a
caveat, and the yield below was checked against tool offsets from 3 to 15 cm.

### Three mistakes worth recording

**The tool offset says where the jaws *are*, not the direction they reach.** `SO101_TOOL` is
(0, −0.0748, 0), and reading that as the reach direction makes the arm able to point its jaws
only at the ceiling — every downward grasp comes back unreachable, and the data looks like the
problem. The same trap had already been hit and documented once in the sibling simulator repo.
`SO101_APPROACH` now states the reach direction and says it was measured.

**The first finger-axis fit used the direction the hand moves into a grasp.** It reads well and
is wrong: over 180 grasps of shoes the pre-grasp motion averaged **116° away from straight
down**, because half of them reach up and over the side of a box first. That fit returned 0.54
consistency and the wrong axis; asking where the fingers *point* at the moment of contact
returns 0.95 and the right one.

**The solver's frame was not the gripper's, and every check lived in the solver's frame.**
`so101()` ends at the wrist_roll joint's frame. The gripper body's frame — the one a recording
of this gripper reports, the one its wrist camera is mounted in — is that frame turned: measured
in the simulator by writing joint angles and reading link positions back, the fingertip frame
sits 98 mm along the terminal frame's **−z**, the servo along −y, and the jaws open along x.
`SO101_APPROACH` said +y, which is right for the body's frame and was "confirmed" against a
recording that reports the body's frame. The tool point sat 75 mm toward the servo. Every plan
and export made this way pointed the real fingers 90° from the human's, and passed every check
— the sweep was just as feasible, the export's forward-kinematics test agreed to the millimetre
— because the checks were all in the same wrong frame. The first policy trained on it scored 0
of 20 on the tomato can before the cause was found by playing its own training data back into
the simulator. The fix is one constant rotation on the way from the recording to the solver
(`SO101_BODY_IN_CHAIN`), the tool point at the jaws, and the rule that a robot's frame is
measured in the simulator against link positions, never against a recording.

Two deployment bugs were found the same way and are recorded in the sibling repo's history:
the simulator's joint action is an offset from a default pose, scaled and clipped to ±28.6°,
which the bridge computed against a pose the scene did not use; and the wrist camera sat low
and flat and showed the policy a white void with no object in it.

---

## What the sweep found

Frames are taken at the rate a policy consumes them (10 Hz, every second frame of the 20 Hz
recording) and one base has to hold a 21-frame window — the policy's observation plus its
action chunk. The grid is 0.42 m across at 0.06 m steps (64 placements), 8 cm above the table.

Two sweeps were run. The first, over all 3,900 candidate episodes, used a robot model whose
gripper frame was 90° off (see "Three mistakes" above); its numbers chose the 3,307 episodes and
are kept below for the record. The second, over exactly those episodes with the corrected frame,
is the one the datasets come from. Solving for the real jaws costs 2.1× the time of the wrong
frame (8.1k against 17.3k solves per second on 16 cores); the corrected sweep took 4.1 h.

Corrected frame, the 3,307 selected episodes plus the 16 VR can episodes:

| task | episodes swept | frames | chunked | one fixed base | episodes ≥90% held |
|---|---|---|---|---|---|
| `Prepare_tableware` | 1418 | 213,289 | **96.2%** | 75.4% | 1244 |
| `get_plate_and_spoon_from_dish_rack` | 524 | 78,348 | **93.1%** | 63.5% | 416 |
| `make_sandwich` | 413 | 62,042 | **53.0%** | 24.4% | 5 |
| `put_shoes_into_storage_box` | 355 | 68,877 | **83.8%** | 41.0% | 78 |
| `take_items_out_of_drawer` | 597 | 106,226 | **88.3%** | 49.5% | 277 |
| `can_v3` (16 VR episodes, `--tcp 0.0748`) | 16 | 1,230 | **77.1%** | 73.1% | 2 |
| **all** | 3323 | 530,012 | **87.4%** | 58.0% | 2022 |

The correction is not uniformly harder. On the same episodes the wrong frame gave the tableware
90.3% / 55.7%, the dish rack 82.7% / 41.7%, the sandwich 90.1% / 48.7%, the shoes 78.8% / 34.6%,
the drawer 90.8% / 58.7% and the can 95.0% / 81.7% (chunked / fixed). Four tasks got easier —
the jaws sit 75 mm further out than the servo did, so the wrist stays further from the table
and the base — and two got much harder: the sandwich and the can come in low and flat, and
aiming real jaws at them costs reach. One fixed base gained as well, so OmniBase's margin over
it narrows from about 36 to 29 points overall. Each task's chunked yield still beats its fixed
one by 20 to 40 points except the can, where 16 short episodes in one spot leave little to gain.

First pass, wrong frame, all candidates (what the selection was made from):

| task | episodes swept | chunked | one fixed base |
|---|---|---|---|
| `Prepare_tableware` | 1500 | 89.3% | 54.9% |
| `get_plate_and_spoon_from_dish_rack` | 600 | 81.5% | 40.9% |
| `make_sandwich` | 600 | 81.7% | 43.4% |
| `put_shoes_into_storage_box` | 600 | 75.5% | 33.3% |
| `take_items_out_of_drawer` | 600 | 90.7% | 58.6% |
| **all** | 3900 | 84.8% | 48.0% |

### What actually limits it

Not reach. **100%** of frames are within 35 cm of *some* candidate base — the arm is long
enough. The limit is the wrist: an SO-101 has five joints, so its tool must point in the
vertical plane through its own base, and a human hand does not. Base placement helps precisely
because moving the base moves that plane, which is why chunking lifts the yield from 58% to 87%.

---

## The dataset

`select` kept every episode that held at least 70% of its frames somewhere (on the first-pass
numbers) and asked for 3,307 of them — **31 GB of the 100 GB budget**, because the yield
threshold turned out to be the real selector and bytes barely discriminate: a FastUMI episode
is 10 MB of video whatever it contains. The 16 VR can episodes were added to both sets, repeated
eight times, so the one scene the policy will be tested in is not a rounding error in its
training data.

Both exports come from the same 3,323 recordings, the corrected plans, the same task string,
the same 10 Hz, the same 512x384 wrist clip. The only difference is where the arm was allowed
to stand.

| | OmniBase (base per window) | one fixed base per episode |
|---|---|---|
| output episodes (contiguous runs the arm can hold) | 9,313 | 5,453 |
| frames at 10 Hz | **349,382** (9.7 h) | 249,902 (6.9 h) |
| size on disk | 1.9 GB | 1.4 GB |
| per-joint step per frame, median / p95 | 1.3° / 10.1° | 1.3° / 10.0° |
| largest joint step in a frame, median / p95 / worst | 4.3° / 16.8° / 60.0° | 4.2° / 16.8° / 60.0° |
| median episode | 3.0 s | 3.8 s |
| export time, 12 processes | 18 min | 11 min |

The fixed-base twin has 72% of the frames. That is less than the 87.4 → 58.0 ratio suggests
because a run also has to be at least 20 frames long to be an episode: frames one fixed base can
hold come in shorter pieces, and the short pieces are the ones that fall out. (On the wrong
frame the twin had 61%; the corrected frame lifted the fixed-base yield more than the chunked
one.)

The motion is smooth at the joint level — a median step of 1.3° per joint per frame at 10 Hz.
The worst step in a typical episode is 17°, which is a hand turning fast seen through a
five-joint arm. Both sets were cut wherever a step exceeded 60°, which is the solver changing
posture between two frames, not the hand. Every exported frame puts the jaws within 15 mm and
20° of where the hand's were; that is checked by forward kinematics on the way out, not assumed,
and the robot model itself is checked against link positions measured in the simulator
(`tests/test_robots.py`).

## Training

The policy is LeRobot's SmolVLA architecture with its vision-language half initialised from
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` — a model that has read web pages and watched video
clips and has never been given a robot action — and its action expert started from random
weights. Nothing in it has seen an SO-101, a UMI gripper, or any robot dataset. The only robot
data it will ever see is the export above.

One run per dataset, identical in every setting but the data:

| | |
|---|---|
| observation | one 512x384 wrist frame, six joint positions in degrees, one task string |
| action | 20-step chunk of joint targets at 10 Hz, 10 executed before re-planning |
| batch / steps | 64 / 32,500 (2.1 M samples: 6 passes over the OmniBase set, 8 over the fixed one) |
| precision | bf16 autocast via Accelerate; LeRobot's own `use_amp` cannot unscale the bf16 VLM's gradients on this stack |
| hardware | one RTX 4070 Ti, 8.5 GB used, 12 loader processes; 0.55 s per step, 116 samples/s, loader wait 5 ms |
| time | about 5.4 h per run, both local; the RunPod budget was not touched |

Two things were measured before the runs rather than assumed. The loader was checked to keep
up — video decode is on the CPU and the GPU never waits for it — and the GPU was found to be
the limit at any batch size from 16 to 96, all within a few percent of the same samples per
second; 64 was chosen because it leaves a third of the memory free and 96 leaves one gigabyte.

The fixed-base run gets the same number of steps, not the same number of epochs, because the
question is what the same compute buys from the same recordings.

TBD: loss curves and final losses.

## The tomato

The object is the YCB tomato soup can the 16 VR episodes were recorded with, in Isaac Sim, from
the wrist camera only, with one task string. Twenty rollouts per row, the arm starting at a
fixed ready pose, success = reached (jaws within 5 cm), lifted, placed in the box.

| policy | scene | reached | lifted | placed |
|---|---|---|---|---|
| OmniBase, plain | can at the wrong-frame position (0.26, -0.06) | 0/20 | 0 | 0 |
| OmniBase, plain | can where the demos grasped it (0.20, -0.01) | 0/20 | 0 | 0 |
| fixed base, plain | same | 0/20 | 0 | 0 |
| OmniBase + 3k steps on the at-home can rows | same | 0/20 | 0 | 0 |
| fixed base + 3k steps on the at-home can rows | same | 0/20 | 0 | 0 |
| OmniBase + 3k steps on the can rows at (0.18, 0.28) | robot fixed at (0.18, 0.28) | TBD | TBD | TBD |
| OmniBase, plain | robot fixed at (0.18, 0.28) | TBD | TBD | TBD |
| fixed base (+ fine-tune) | robot fixed at (0.18, 0.28) | TBD | TBD | TBD |

Why the zeros, measured rather than guessed (the full trail is in
`docs/experiment/2026-09-09_fastumi_so101_smolvla.md`): the 16 demonstrations were made by a
hand that approached the can from the side, with its approach axis 40-118 degrees off the
bearing from the robot's base. A five-joint arm's tool must lie in the vertical plane through
its base, so from where the robot was placed **none** of the demo frames are executable at
15 mm / 20 degrees -- `omnibase sweep --home` says 0.0%, and the base it moves to instead is
25-30 cm to the left. Retargeting the demos "at home" anyway, at a 90-degree orientation
tolerance, yields contorted postures that no rollout starts from; a policy fine-tuned on them
closes its jaws in the air. What OmniBase tells you here is the honest thing: put the robot
where the demonstrations are reachable. That is the last row.

---

## Cost

| | |
|---|---|
| downloaded | 31 GB of a 100 GB budget |
| local GPU | RTX 4070 Ti, 12 GB |
| rented GPU | $0 of a $10 budget |

## What this does not show

TBD
