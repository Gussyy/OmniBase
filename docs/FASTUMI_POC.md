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

### Two mistakes worth recording

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

---

## What the sweep found

Frames are taken at the rate a policy consumes them (10 Hz, every second frame of the 20 Hz
recording) and one base has to hold a 21-frame window — the policy's observation plus its
action chunk. The first task was swept at 1,500 episodes on a 289-placement grid (63 min on 16
cores); the rest at 600 episodes on a 225-placement grid, because the selection budget holds
about 9,000 episodes and 3,900 candidates against it is already a choice.

| task | episodes swept | frames | chunked | one fixed base | episodes ≥90% held |
|---|---|---|---|---|---|
| `Prepare_tableware` | 1500 | 225,545 | **89.3%** | 54.9% | 728 |
| `get_plate_and_spoon_from_dish_rack` | 600 | 89,291 | **81.5%** | 40.9% | 37 |
| `make_sandwich` | 600 | 88,731 | **81.7%** | 43.4% | 254 |
| `put_shoes_into_storage_box` | 600 | 114,273 | **75.5%** | 33.3% | 2 |
| `take_items_out_of_drawer` | 600 | 106,752 | **90.7%** | 58.6% | 340 |
| **all** | 3900 | 624,592 | **84.8%** | 48.0% | 1361 |

Pilot, 40 episodes of `put_shoes_into_storage_box`: **80.0% chunked against 39.9% fixed.**
Pilot, 12 episodes of `Prepare_tableware`: **92.7% chunked against 62.6% fixed.**

### What actually limits it

Not reach. **100%** of frames are within 35 cm of *some* candidate base — the arm is long
enough. The limit is the wrist: an SO-101 has five joints, so its tool must point in the
vertical plane through its own base, and a human hand does not. Base placement helps precisely
because moving the base moves that plane, which is why chunking roughly doubles the yield.

---

## The dataset

`select` kept every episode that held at least 70% of its frames somewhere and asked for
3,307 of them — **31 GB of the 100 GB budget**, because the yield threshold turned out to be the
real selector and bytes barely discriminate: a FastUMI episode is 10 MB of video whatever it
contains. The 16 VR can episodes were added to both sets, repeated eight times, so the one scene
the policy will be tested in is not a rounding error in its training data.

Both exports come from the same 3,323 recordings, the same task string, the same 10 Hz, the same
512x384 wrist clip. The only difference is where the arm was allowed to stand.

| | OmniBase (base per window) | one fixed base per episode |
|---|---|---|
| output episodes (contiguous runs the arm can hold) | 8,463 | 4,427 |
| frames at 10 Hz | **256,433** (7.1 h) | 155,945 (4.3 h) |
| size on disk | 1.4 GB | 815 MB |
| per-joint step per frame, median / p95 | 1.5° / 9.3° | 1.3° / 8.7° |
| largest joint step in a frame, median / p95 / worst | 4.3° / 15.2° / 59.7° | 4.0° / 14.1° / 59.3° |
| export time, 12 processes | 17 min | 9 min |

The fixed-base twin has 61% of the frames. That is less than the 84.8 → 48.0 ratio suggests
because a run also has to be at least 20 frames long to be an episode: frames one fixed base can
hold come in shorter pieces, and the short pieces are the ones that fall out.

The motion is smooth at the joint level — a median step of under 2° per joint per frame at 10 Hz,
and the wrist roll, the one free joint the solver re-aims, reverses direction in under 1% of
frames. The worst step in a typical episode is 16°, which is a hand turning fast seen through a
five-joint arm. Both sets were cut wherever a step exceeded 60°, which is the solver unwinding a
wrist-roll limit, not the hand.
Every exported frame puts the gripper within 15 mm and 20° of where the
hand was; that is checked by forward kinematics on the way out, not assumed.

## Training

TBD

## The tomato

TBD

---

## Cost

| | |
|---|---|
| downloaded | TBD of a 100 GB budget |
| local GPU | RTX 4070 Ti, 12 GB |
| rented GPU | TBD of a $10 budget |

## What this does not show

TBD
