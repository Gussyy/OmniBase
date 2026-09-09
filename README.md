# OmniBase

**Where would the robot have to stand?**

A person demonstrating a task with a hand-held gripper does not think about a robot's reach.
Replay what they flew against a fixed-base arm and much of it is simply unreachable — not
because the arm is slow or the controller is bad, but because no joint angles exist.

OmniBase stops treating the base pose as a property of the robot. In a recording it is a **free
variable**: nothing in the data says where the robot was standing, because there was no robot.
So choose it — and choose it again whenever it stops working.

numpy and scipy. No simulator, no learned model, no rollout.

---

![OmniBase: recorded wrist camera on the left, the same data driving two SO-101s on the right](docs/omnibase_demo.gif)

**Left: the data.** The wrist camera of a hand-held gripper, one episode, exactly as recorded.
**Right: the same data, retargeted.** Two SO-ARM101s driven to joint angles OmniBase solved for,
each standing where it says a base would have to stand. Frame-for-frame in sync — the recording
never changes, only the robot under it does. The base is re-placed three times, and across the
episode the arms hold 100% of frames where the best single fixed base holds 66%.

The render is Isaac Sim, and it is only a picture: OmniBase computed the placements and the
joint angles with the arithmetic in this repository, and never opened a simulator to do it.
[Full-resolution clip](docs/omnibase_demo.mp4). Both panels are at their native 480x360 — the
recording's own wrist-camera resolution — so nothing here has been upscaled into looking
sharper or blurrier than it is.

---

## The idea in one table

A policy never sees a whole episode at once. It sees an observation window and predicts an
action chunk — a couple of dozen frames. A base only has to be right for that long.

Measured on 16 episodes of a two-handed can-picking recording, retargeted to an SO-ARM101:

| one base must serve | right hand usable | left hand usable |
|---|---|---|
| the whole episode | 56% | 50% |
| 250 frames | 63% | 58% |
| 100 frames | 80% | 80% |
| 50 frames | 93% | 95% |
| **21 frames** (obs + action chunk) | **99.4%** | **100%** |

Shortening the commitment from "one base for the episode" to "one base per training window"
took the same recordings from roughly half usable to essentially all of it. Across all 32
episode/hand pairs, chunked retargeting held **100%** of frames; the best single fixed base
managed 61–100% depending on the episode.

Moving the base once already pays before any chunking. At the placement the teleop config
happened to use, 18% and 24% of frames were reachable. At the best single placement — 12 cm
back and 8 cm up — 76% and 73%.

---

## Install

```bash
pip install -e .
```

Requires Python 3.9+, numpy and scipy. `pandas` and `pyarrow` are only needed if you load
LeRobot datasets through `omnibase.load`.

---

## Use it on your data

The library only ever needs two arrays per hand: `pos` (N, 3) in metres and `quat` (N, 4) as
`(x, y, z, w)`, both in one fixed world frame. If you have those, everything below works.

```python
import omnibase as ob

chain = ob.so101()                                    # or build your own, see below
ep    = ob.load("datasets/mine", episode=6)           # LeRobot v2.1 or v3.0
pos, quat = ep.hands[0].pos, ep.hands[0].quat

cells  = ob.base_grid(span=0.42, step=0.02, height=0.08)
F      = ob.feasibility(chain, pos, quat, cells)
chunks, held = ob.chunk([F], cells, window=21)        # window = obs buffer + action chunk

print(f"{held.mean():.0%} of frames held across {len(chunks)} chunk(s)")
print("best single base: %.0f%%" % (100 * ob.best_fixed(F, cells)[1]))
for c in chunks:
    print(f"  frames {c.start:4d}-{c.stop:4d}  base {c.bases[0].round(3)}  held {c.held[0]:.0%}")
```

Pass several hands at once and they share one set of chunk boundaries — they are in one
recording, and a cut that lands mid-reach for the other hand is not a cut you can use:

```python
Fs = [ob.feasibility(chain, h.pos, h.quat, cells) for h in ep.hands]
chunks, held = ob.chunk(Fs, cells, window=21)
```

### Arms that cannot be placed separately

The above places each arm on its own. That is right for arms on separate stands and wrong for
almost every real bimanual robot: two arms on one torso, a humanoid, a pair bolted to the same
rail. There the spacing is hardware, and the only thing you choose is where the whole assembly
stands — so a placement that suits the right arm is no use if it strands the left one.

Place the assembly instead:

```python
mount = ob.pair(0.46)                                  # two arms, 0.46 m apart, one plate
cells = ob.mount_grid(span=0.34, step=0.02, height=0.08,
                      yaws=np.radians([-20, 0, 20]))   # a torso can turn, too
combined, per_arm = ob.mount_feasibility(chain, [(h.pos, h.quat) for h in ep.hands],
                                         mount, cells)
chunks, held = ob.chunk([combined], cells, window=21)  # one placement to choose, not two

for (p, rot) in ob.arm_bases(mount, chunks[0].bases[0]):
    print("bolt an arm at", p.round(3))
```

`combined[f, c]` is true only where **every** arm can hold frame `f`. For arms mounted at an
angle — shoulders usually are — give each offset as `(x, y, z, qx, qy, qz, qw)` and build the
`Mount` directly instead of using `pair`.

```bash
python -m omnibase plan datasets/mine --episode 6 --pair 0.46 --mount-yaw=-20,0,20
```
```
  rigid pair, 0.460 m apart -- placing the assembly, not the arms

  5 chunk(s); the assembly moves 4 time(s):
    chunk 1: frames   21-  72   mount (-0.05,+0.05,+0.08) yaw +0
                                -> right [-0.05, -0.18, 0.08]  left [-0.05, 0.28, 0.08]
                                both held 100.0%

  BOTH arms at once: chunked 71.1% of frames held, against 41.4% for the best single placement
     right alone would manage  45.4% -- bolting them together costs the difference
      left alone would manage 100.0% -- bolting them together costs the difference
```

Note what that last pair of lines is for. Coupling is a constraint and can only ever cost you
reach, so it is worth seeing the size of the bill: here the left arm could have had everything
and gives most of it up to stay bolted to the right one. If that number is uncomfortable, it is
an argument about your hardware, not about your data.

Mount yaw is a real degree of freedom in a way a single arm's is not: turning one arm about its
own base is something its shoulder joint absorbs, but turning a torso swings the far arm through
an arc and changes what it can reach.

### Your robot already has a base

Everything above treats the placement as free. It is free in the *data* — but if you own the
robot, one placement is worth more than the others: the one it actually stands at. Frames
retargeted there need no explanation at deployment. Frames retargeted somewhere invented are
still training data, but they describe a robot standing where yours does not.

So say where it stands, and the real base is used wherever it works and left only where it
cannot — returning at the first frame it can serve again:

```python
chunks, held = ob.chunk([F], cells, window=21, home=(0.10, -0.04, 0.08))
print(ob.home_share(chunks))            # fraction of frames served from the real base
for c in chunks:
    print(c.start, c.stop, "home" if c.at_home[0] else "moved", c.bases[0].round(3))
```

```bash
python -m omnibase plan datasets/mine --episode 6 --pair 0.46 --home=-0.10,0.06,0.08
```

On the two-handed can data with the arms rigidly coupled, that is free improvement in both
directions at once:

| | frames held | at the real base |
|---|---|---|
| placement free to wander | 43.0% | 0% |
| **home first** | **46.2%** | **20.9%** |

It holds *more* while standing in the right place *more often*, because preferring home is a
tie-break rather than a handicap — a test pins that it can never score below simply standing
still. `Chunk.at_home` marks which chunks are which, so you can weight or filter on it when you
build a training set.

Coming back promptly is the part that matters. Only re-checking home when the stand-in fails
leaves the robot parked somewhere invented long after it could have gone back; here that was
the difference between 0% and 20.9%.

### LeRobot datasets

Both on-disk layouts are read, and both kinds of action column:

| | |
|---|---|
| **v2.1** | one parquet per episode, `data/chunk-000/episode_000000.parquet` |
| **v3.0** | episodes concatenated, `data/chunk-000/file-000.parquet`, sliced by `episode_index` |
| **end-effector poses** | columns named `<hand>_x, _y, _z, _qx, _qy, _qz, _qw` (+ optional gripper) |
| **joint angles** | columns named after a robot's joints — pass `chain=` and they are run through forward kinematics |

Columns are matched by the names in `meta/info.json`, not by position, so three hands, or a
gripper column in an odd place, or no gripper at all, all read correctly. Check what a dataset
holds before planning against it:

```bash
python -m omnibase data datasets/mine
```
```
  codebase v3.0  fps 30  episodes 16
  action   shape [16]  names ['right_x', 'right_y', ... 'left_qw', 'left_jaw']
  -> hand 'right': 249 frames, pose, gripper column found
  -> hand 'left':  249 frames, pose, gripper column found
```

Joint-space datasets are readable but are usually the *wrong input*: they came off a robot that
already had a base, so there is nothing to place, and OmniBase will correctly report 100%
reachable at every window length. They are useful for asking where that robot *should* have
stood, or for treating one robot's recording as a source for a different arm.

If your data is not LeRobot at all, skip the loader — the library only ever needs `pos` (N, 3)
and `quat` (N, 4) as `(x, y, z, w)` in one fixed world frame.

### Command line

```bash
python -m omnibase data  datasets/mine                  # what is in here, and can it be used?
python -m omnibase plan  datasets/mine --episode 6 --out plan.json
python -m omnibase curve datasets/mine --episode 6      # is this idea worth anything on my data?
python -m omnibase map   datasets/mine --episode 6 --frames 0:112
python -m omnibase robot so101                          # check the arm reads right
python -m omnibase sweep datasets/mine --workers 14 --out plans.json   # the whole dataset
python -m omnibase level datasets/mine --frame fastumi --out level.json  # what do its numbers mean?
python -m omnibase select plans.json --budget-gb 90 --out selection.json # what is worth fetching?
python -m omnibase export plans.json --out datasets/so101 --fps 10       # make it trainable
```

`--hands` takes names or indices (`--hands right`, `--hands 0,1`); the default is every hand in
the episode. `--column observation.state` reads what was measured instead of what was
commanded.

---

## A whole dataset

`plan` does one episode. `sweep` does all of them, across as many cores as you have:

```bash
python -m omnibase sweep datasets/mine --workers 14 --out plans.json
```

```
16 episodes x 484 placements, 14 worker(s)

16 episodes, 3674 frames, 3556432 solves in 146.5s (24.3k solves/s)
  chunked 99.9% of frames held, against 82.6% for the best single fixed base
```

The parallelism is here, over episodes, rather than inside the reachability sweep. Episodes are
independent and there are many of them, so splitting at this level fills every core with no
coordination, pays the process pool's startup once for the run instead of once per episode, and
lets each worker keep the reach envelope it built. Splitting the *cell* sweep instead was
measured at the same throughput per core while rebuilding a pool for every call.

`--episodes 0,1,2` limits which ones. Everything else — `--window`, `--home`, `--pos-tol`,
`--step` — means what it means for `plan`. One episode failing to load is reported and the rest
still run.

### How fast

Reachability is the whole cost: one inverse-kinematics solve per (placement, frame) pair, and a
1764-cell grid over 100 episodes is about 44 million of them. Two things carry it, and they
multiply because one removes work and the other makes what is left cheaper.

**Most of a sweep is settled by arithmetic.** Only about 6% of pairs are reachable — the grid is
a metre across and the arm is not — so the great majority of the runtime used to go into proving
the obvious with a full iterative solve. `Chain.cannot_reach` applies necessary conditions
derived from the chain itself (a reach envelope as a solid of revolution about joint 0, the same
test for points out along the tool axis, and one scalar invariant that catches a gripper held
sideways to the arm's plane) and rejects 70–85% of pairs in about 15 ms per 56,000. Every
threshold is widened by a bound on how far the truth can sit from the nearest sample, so each
region is a *superset* of the real one and being outside it is proof, not a guess.

**The solver itself got cheaper**, without changing a step it takes: the rotation matrices are
written out instead of stacked, the pseudo-inverses are applied to vectors rather than formed,
the forward-kinematics prefix is hoisted out of the free-joint sweep, and all three seeds run as
one batch.

Measured on this machine, per-episode:

| grid | before | after | |
|---|---|---|---|
| 225 placements x 249 frames | 82.1 s | 11.7 s | **7.0x** |
| 484 placements x 249 frames | 218.6 s | 23.0 s | **9.5x** |
| 225 placements x 251 frames | 82.5 s | 4.4 s | **19.0x** |

It pays off more on bigger grids, because a bigger grid is mostly further away. Across a dataset
with `--workers 14`, 0.7k solves/s becomes 24.3k — so the 44-million-solve run above is about
half an hour rather than the better part of a day.

**None of it changes an answer.** The optimisations were kept only where the feasibility mask
came out bit-identical: 289,041 mask entries across three grids and two episodes, zero differing.
`test_pruning_agrees_with_solving_everything` pins that in the suite by solving a grid both ways,
and `test_pruning_only_rejects_the_truly_unreachable` takes poses the arm demonstrably just held
and requires that none is rejected. A faster library that quietly reports an arm cannot reach
something it can would be worse than a slow one.

Two things were measured and *not* shipped, in case you were about to try them. Warm-starting
the solver from the neighbouring cell is 2.6–8.4x faster and moves 0.2–1.1% of the mask, always
reporting less reach than there is. Cramer's rule for the inner 3x3 solve is the same speed as
the Cholesky now used and looks identical on a good day, but its backward error is nine orders
worse and near-singular it returns `nan`, which reads downstream as "cannot reach".

---

## Recordings that do not say what their numbers mean

Everything above assumes you can hand this library `pos` and `quat` in one fixed world frame. A
hand-held rig often cannot. FastUMI's poses, for instance, are relative to the tracker's own
pose at frame 0, so nothing in the file says which way is up; the tracker sits somewhere on the
gripper nobody wrote down; its three Euler angles are in one of twelve conventions and the file
does not say which. Every one of those is wrong by default, and a wrong one does not fail — it
reports that your data is unreachable.

`omnibase level` measures all four from the recording. Run it first, on anything new:

```bash
python -m omnibase level datasets/mine --frame fastumi --out level.json
python -m omnibase sweep datasets/mine --level level.json --stride 2 --out plans.json
```

```
  euler xyz, tool 0.06 m ahead of what the rig tracked
  up [0.993, -0.074, 0.096] in the recording's own axes
  360 contacts from 60 episodes lie flat to 22 mm rms (over the 80% kept)
  2.67% of all frames end up under the table
  fingers along [-0.32, 0.013, -0.947] of the rig's own axes, and they point at the table at
  14 deg when they close
  the same body direction does that at every grasp to 0.95 of 1.00
  looks usable.
```

Each number comes from the part of the data that can see it:

| | measured from |
|---|---|
| **up** | the frames where the jaw changed state — the hand touching things that were standing on a surface. Positions only, so no convention can corrupt it. Levelled per episode, because a rig docked in the same slot every take shares its orientation between takes and not its origin. |
| **the sign of up** | whichever choice leaves less of the recording underground. Asking the jaws instead reads well and is wrong the moment a task grasps from the side, which taking a plate out of a rack does. |
| **the finger axis** | the body direction that points at that surface whenever the jaws close. |
| **the Euler convention** | the same consistency. Read three angles in the wrong order and each frame's body frame is turned differently, so no single direction can point at the table at every grasp. |

That last statistic — `aligned` — is the one to watch, because it tests the whole story at once.
It reaches 1.0 only if the episodes really do share a frame, the angles really were read in the
right order, and the fitted table really is the table. Across FastUMI's five pick-and-place
tasks it reads **0.89 to 0.99**, with the fingers 6 to 22 degrees off the table at a grasp —
which is never fitted, and is what a person picking something up actually does.

**The tool offset is the one thing that stays loose**, and the library says so rather than
hiding it. It is found by asking which offset makes grasps taken at different wrist angles land
on the same surface, which works when the objects are one height and stops working when they
are not: a grasp happens at table height plus half an object, so a box of assorted things writes
5 cm of scatter over an effect worth about 1. `test_the_tool_offset_is_only_as_sharp_as_the_objects`
pins both halves. Check that your answer does not depend on it.

---

## From a plan to a dataset

A plan is an answer about a recording. A policy cannot train on an answer, so `export` hands the
chosen placement back to the solver — this time keeping the joint angles `feasibility` throws
away — and writes an ordinary LeRobot v3.0 dataset: joint angles in degrees, the wrist video cut
to match, and the metadata LeRobot reads without conversion.

```bash
python -m omnibase select plans_*.json --budget-gb 90 --out selection.json
python -m omnibase export plans_*.json --out datasets/so101 --fps 10 --workers 8
python -m omnibase export plans_*.json --out datasets/so101_fixed --fixed-base --fps 10
```

`select` exists because of the shape of this data: a FastUMI episode is 30 kB of poses and 10 MB
of video, and every question OmniBase asks is answered by the 30 kB. Sweep everything, decide,
and fetch video only for what survived. The last line is the honest comparison — the same
episodes with the arm bolted down in one place, keeping only the frames it can hold.

Two things the export had to learn, and both are about the difference between counting what an
arm can reach and writing down what it should do.

**A redundant joint has several ways to make the same pose.** Solving each frame independently
lets it swap between them; measured on FastUMI, a 320-degree step in wrist_roll between one
frame and the next in an episode where nothing else moved more than fifteen. `solve` re-solves
such a frame from the previous frame's answer, and re-aims a free joint — a wrist roll, which
moves the tool point nowhere — to the nearest value that still holds the tolerance.

**Some of those steps are real, and the answer is to stop.** The SO-101's wrist stops at −157
and +163 degrees, so a hand that rotates past the gap leaves the arm to unwind 320 the other
way. Both frames either side are honest; the line between them is not. `--max-step` ends the
episode there — on a slice of FastUMI's tableware task that costs 10 frames of 1381 and takes
the worst step from 320 degrees to 34.

The export also notes where the lens's picture sits — a fisheye leaves its corners dark — and
writes the ellipse into `meta/omnibase.json`, so a simulator rendering a full rectangle can be
masked to the same shape instead of somebody eyeballing it later.

---

## Where should it stand, and where should it not

Reachability alone is a poor answer. A base can reach every frame and still be a bad place to
put a robot: if the arm is stretched flat, or riding a joint stop, or passing through a
singularity, the demonstrations you generate there are ones a policy will struggle to imitate
and a real arm will struggle to execute. Those placements quietly produce bad data.

`score_map` scores every candidate on all of it, from the joint angles actually solved for — so
a placement that only reaches by jamming a joint against its stop scores badly even though it
"reaches":

| term | meaning | good |
|---|---|---|
| `coverage` | fraction of frames the arm can hold at all | high |
| `limit_margin` | how far joints stay from their stops (1 = mid-range, 0 = on the stop) | high |
| `manipulability` | distance from singularity; low = stretched flat or folded | high |
| `rot_margin` | how much of the orientation tolerance is left unspent | high |

Each cell gets a verdict — `good`, `workable`, `marginal`, `unusable` — and the map draws in the
terminal, because a placement you cannot see is a number you cannot check:

```
  score: blank = cannot reach, '@' = best (0.537)
  x -0.42 .. +0.42 m down, y -0.42 .. +0.42 m across
  +0.30 |         :
  +0.24 | --:      +:
  +0.18 | ++=:. :-==
  +0.12 | ***+- :*#+-
  +0.06 | +#*+= .:**
  +0.00 | -#*#=    :
  -0.06 | =#@%=::
  -0.12 | +#%#=-
  -0.18 | =*#==-
  -0.24 |  ===
```

```python
smap = ob.score_map(chain, pos, quat, cells)
cell, i, report = ob.best_spot(smap)
print(report)
print(ob.ascii_map(smap, "coverage"))
```

The single most useful output is often not the converted data at all — it is a **mounting
recommendation**. Where the density peaks is where to bolt the arm, and you act on that with a
drill.

---

## Your own robot

A dozen lines. Read the joint origins, rotations, axes and stops out of whatever describes your
arm and hand them over. Both constructors land in the same place:

```python
from omnibase.chain import Chain, Joint

joints = [
    # exactly what a USD PhysicsRevoluteJoint carries: quaternion (w, x, y, z), limits in degrees
    Joint.from_usd(origin, local_rot_wxyz, "z", -110, 110, "shoulder_pan"),
    # or exactly what a URDF joint carries: origin xyz/rpy, an arbitrary axis, limits in radians
    Joint.from_urdf(xyz, rpy, (0, 0, 1), -1.92, 1.92, "shoulder_pan"),
    ...
]
chain = Chain(joints, tool=(0.0, -0.0748, 0.0), name="mine")
print(omnibase.describe(chain))
```

`tool` is the point **between the fingers**, not the wrist flange. Getting it wrong shifts every
reachability answer by however far out it is.

**Check it before you trust it.** A chain with a plausible-looking wrong number does not fail,
it quietly reports that your data is unreachable. `tests/test_omnibase.py` round-trips random
joint configurations through forward and inverse kinematics; do at least that much for a new
arm, and if you have the robot or a simulator, compare against poses it reports for known joint
angles.

---

## What this does not do

- **It cannot buy a degree of freedom.** Five joints trap the approach axis in the plane the
  shoulder picks. Tighten the heading tolerance to 10° and yield collapses to about 50%
  wherever the base stands. Base placement moves *where* the deficit bites, not whether.

- **Many placements is not many demonstrations.** A window with 131 valid base cells has a
  contiguous feasible *region*, roughly 520 cm² of table — not 131 independent samples.
  Sampling it densely mostly duplicates.

- **The tolerance is an assumption.** Everything here rests on "15 mm and 20° still grasps",
  and that single threshold moves the answer between 50% and 85%. What actually breaks a grasp
  is worth measuring once, on your gripper and your object. It is a calibration, not a
  per-episode verification.

- **It assumes your observations are base-invariant.** Re-placing the base is free because a
  wrist camera rides the gripper and sees the same thing wherever the robot stands. Add a
  camera that watches the scene from the table and it sees the base move, and the trick dies.

- **The prune is verified, not proven.** The bounds that widen each region are derived
  honestly but by hand, and the sample grid that builds them is a grid. The evidence is
  empirical and large — every pruned pair on a real sweep re-solved as unreachable, and the
  masks are bit-identical to solving everything — but it is evidence, not a proof. If you add
  an arm shaped very differently from the ones here, run `test_pruning_only_rejects_the_truly_unreachable`
  against it before trusting a sweep.

- **The solver is general, so it is slightly conservative.** Damped least squares with a
  task-priority split — position exactly, orientation in the null space of what is left — plus
  an exhaustive sweep of any joint that moves no position at all (a wrist roll through the
  grasp point, typically). Checked against a hand-derived closed form for the SO-101 on real
  recordings, it reports position exactly and orientation within ~1.5°, and calls 0–2.4
  percentage points *fewer* frames reachable. That is the safe direction to be wrong in: it
  never claims a frame is usable when it is not. About 3.5% of exactly-achievable poses land in
  the wrong branch, which is where that gap comes from.

---

## Tests

```bash
python tests/test_omnibase.py       # kinematics, chunking, scoring
python tests/test_data.py           # LeRobot loading -- builds its own datasets in a temp dir
# or, both:  python -m pytest
```

The ones that matter are the two that could pass while being wrong: inverse kinematics that
reports success it did not achieve, and a chunker that claims a placement holds frames it does
not. Both are checked against forward kinematics rather than against themselves.

---

## Licence

MIT.
