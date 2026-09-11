# Jeff's notebook

Running notes to myself: ideas, half-done math, things to check, things that turned out wrong.
Not a report. The experiment log (`docs/experiment/`) is where the settled findings go; this is
where they come from. Newest entries at the bottom of each day.

---

## 2026-09-11

### Where we stand (so I stop re-deriving it)
- Proven: OmniBase makes human demos executable (0% -> 45% at the deployment base; 87% vs 58%
  yield; 2x grasp events). Measured, not statistical.
- Suggested, not proven: better policy (15 vs 2 lifts, p=0.0007) -- confounded by +40% frames and
  one seed. Frame-matched control prepared in `E:/data/fastumi/plans_matched/`, ~6.5 h, not run.
- Failed as built: multi-base augmentation in joint space (actions average across bases -> jaws
  3.3 cm off). Fix = base in the state, or relative-EE actions.
- Both policies are goal-position-locked: dead beyond ~3 cm object shift, ignore distractors and
  the prompt (single task string -> prompt-blind by construction).

### Idea: base-shift eval (user's, 2026-09-11 afternoon)
Keep the scene, move the robot base by d, run the policy. Base shift == object shift from the
policy's point of view, *plus* the box and table shift in the image. Bracket it with two
references replayed from the SAME rows:
- fixed: rows solved for base 0, replayed from base d  -> misses by ~d (pure replay)
- re-solved: rows re-solved by OmniBase for base d     -> lands on the can (pure generaliser)
A policy's curve between them says how much it learned the object vs the motion. Feasibility of
the re-solve gives the fair-test boundary for free.

Cost: 17 offsets x 1 Isaac boot (~1 min) -- an hour. Cheaper than any object grid.

### What actually happened
1. Pipeline `E:/data/out/pipeline_baseshift.sh`, `shift_plan.py`, patched `grasp_check.py`
   (--root list, --base-offset, --out jsonl, --dwell, --frames, --d_close).
2. Fixed-replay d_close grows linearly with the shift (y-6: 6.2 cm, y-8: 8.1, x+8: 8.2). Good:
   that reference behaves.
3. Re-solved d_close stays small where feasible (y-6: 2.0 cm) but retained frames fall off fast
   (y-8: 39%, x+8: 23%). Also good: the boundary is real and OmniBase reports it.
4. **Nothing lifts. Not even at d=0.** 0/7 across the board; the only two lifts were fixed
   replays at y+2/y+4 (ep2), i.e. luck.
5. Diagnosis so far:
   - The demos (VR gripper) close and lift in the same instant. The sim jaws need ~0.6 s (6 rows)
     to reach a 68 mm can; by then the arm is 10-19 cm up. --dwell 8 at the closing row fixes
     the timing: joint reaches -0.0355 of -0.036 within 2 rows.
   - Still 0 lifts with the dwell. Jaws close fully 1.1-1.5 cm from the can centre and the can
     does not move a millimetre ("can xy moved 0.0"). The fingers are not where the FK says.
   - Probe at ep5's closing row: FK tool point (0.206,-0.003,0.058); gripper_frame_link body at
     (0.218, 0.007, 0.036) -- 2.3 cm off in xy, 2.2 cm lower. No body matched
     grip/jaw/finger/pad/tip except gripper_base/gear/frame_link -> the finger links are named
     something else. Print all bodies next.
   - Note the policy (closed-loop, same rows in training) DID lift 4/4 here. So the scene is
     graspable; the open-loop replay is the thing that fails. Candidates: (a) tool-point offset
     wrong for this asset by ~2 cm, (b) fingers closing beside the can (jaw axis direction),
     (c) something with the gripper_gear joint (first of three 'grip' joints stays at 0.0006).

### Consequences for the eval idea
- Lift is a bad reference metric while the open-loop replay can't grasp. Jaw-to-can distance at
  the closing row (d_close) is the honest one and it already separates the two references.
- If the replay can't be made to lift, the policy's 4/4 was closed-loop correction on top of a
  memorised approach: it goes to the trained spot and closes when the can is between the jaws.
  That is consistent with "lifts up to 3 cm shift, none beyond" (jaw span ~ +-3 cm).

### To check
- [ ] All body names + positions at the closing row; where are the finger pads really?
- [ ] Is SO101_TOOL (0,0,-0.0748) the pad centre for so101_full in the USD, or the URDF's?
- [ ] Try the replay with the jaws commanded shut 2-3 rows EARLY (the demo's own timing was
      instantaneous, so the arm was already at the can at the close row, not before).
- [ ] Rerun the 17 offsets with --dwell and d_close once the replay lifts at d=0.
- [ ] Six offsets (x-2..x-8, y+6, y+8) never ran: stray Isaac processes from the `app.close()`
      AttributeError held the GPU. Fixed (`app.app.close()`), killed strays.

### Math scratch
- Jaw span 128 mm open, 36 mm travel per finger -> 56 mm gap closed. Can 68 mm -> 6 mm squeeze
  per side if centred. Off-centre by e along the jaw axis: one finger short by e, other pushes
  e deeper; contact on both sides needs the can to slide e. At 1.3 cm it should still contact.
  It does not. So the geometry assumption is wrong, not the tolerance.

### 19:50 -- E:\data\out deleted by the user
Everything under it is gone: datasets, checkpoints, results folders, eval JSONs, AND every pipeline
script (they were never in git -- my mistake). Survivors: this repo, so101-scene (configs, can_v3d,
patched grasp_check.py), E:\data\fastumi (raw + plans2 + plans_matched), the experiment log's
tables. Every lost script is in the session transcript
(`C:\Users\Admin\.claude\projects\E--work-Isaacsim-OmniBase\dbf4e509-...jsonl`) if ever needed.
Lesson: scripts go in `scripts/experiment/` in the repo from now on; data in E:\data\<name>.
Base-shift work dir is now `E:\data\baseshift` (tiny: 17 exports of 7 chunks).

### 19:55 -- reframe from the user: math only, no simulation, no new data. Eval without rollouts.
The library is about robot DATA. Think data science. Ideas, in the order I'd build them:

**1. Offline base-shift probe (do this now).**
The wrist camera is attached to the hand, so the image at a given tool pose is the same from any
base. Only the joint state differs, and OmniBase can re-solve it for any base b:
    state_b(t) = IK(pose(t) - b),   action_b(t) = state_b(t+1)
Feed a policy (image(t), state_b(t)) and compare its prediction to action_b(t) -- in WORLD space
through FK, so the number is centimetres at the tool point and base-free:
    err(b) = | FK(pred) - FK(action_b) |
A memoriser predicts action_0 whatever the state: err(b) ~ |b|. A generaliser: flat. No rollout,
no renderer, no new data; the labels come from the same demos re-solved by math.
References that need no training: replay (returns action_0 by frame index) and oracle (action_b).
Cheap learned probes: a kNN / small MLP on state->action trained at base 0 vs trained on a base
grid. The second should pass; the first should fail by ~|b|. That is the multi-base augmentation
question answered in minutes instead of a fine-tune plus a grid of rollouts.
Validation of the metric itself: the sim pipeline (still running) gives jaw-to-can distance at
the closing row for the same offsets; the offline err(b) should track it.

**2. Action ambiguity given the observation (data-only diagnostic).**
The multi-base failure was: same image, different joint actions -> the model averages them ->
jaws 3.3 cm off. That is measurable BEFORE training: for frames with (near-)identical tool pose,
the spread of the joint action across bases, mapped through FK to centimetres. Call it
ambiguity(b-set). Zero for a relative-EE action space, large for joint space with many bases.
It tells you whether an augmentation will help or hurt a given action space. Nobody ships this.

**3. Relative-EE export.** Removes the base from the action; ambiguity -> 0 by construction.

**4. Dataset diagnostics as first-class outputs** ("robot data science"): frames executable per
robot per base (the map), grasp-event count, workspace/joint coverage, per-episode yield, what a
fixed base throws away vs OmniBase keeps, duplicate/near-duplicate frames, hold/pause rows
(drop_holds found 47% of the VR rows were pauses). All math on the dataset.

**5. Offline policy scoring on held-out demos** (standard, but with OmniBase labels): the
policy's world-frame action error on demos it never saw, at the deployment base. Rollout-free
success proxy; correlate with the sim once, then never need the sim again.

Math for 1: with base offset b and joint solution q_b, FK(q_b) + b = pose (world), so
FK(action_b) = pose(t+1) - b for every b. err(b) = |FK(pred_b) - (pose(t+1) - b)|. For the replay
reference pred_b = action_0 so err = |pose(t+1) - 0 - pose(t+1) + b| = |b|. Exactly linear.

### 20:20 -- offline probe ran. `scripts/experiment/offline_probe.py`, ~1 min, no Isaac.
Median world-frame miss of the predicted next tool point (cm), 7 can chunks, 287 frames at base 0:

| policy | 0 | 2 cm | 4 cm | 6 cm | 8 cm |
|---|---|---|---|---|---|
| replay (base-0 rows) | 0 | 2.0 | 4.0 | 6.0 | 8.0 |
| relative joints (base-0 deltas) | 0 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.3-0.8 |
| kNN, base 0 | 0 | 2.0 | 3.6-4.0 | 4.2-6.0 | 5.1-7.7 |
| MLP, base 0 | 0.4 | 1.6-2.3 | 2.7-4.0 | 3.5-6.4 | 4.4-9.7 |
| kNN, 5x5 base grid (+-6) | 0 | 1.0-1.7 | 1.0-2.0 | 0.0 | 2.0-3.4 |
| MLP, 5x5 base grid (+-6) | 1.4 | 1.3 | 1.2 | 1.3 | 1.6-1.9 |
| oracle (re-solved) | 0 | 0 | 0 | 0 | 0 |

Reading it:
- replay = |b| exactly: the metric is calibrated in centimetres, by construction.
- A state-conditioned policy trained at ONE base is a memoriser under base shift (kNN and MLP
  both track replay). Trained on a base grid it is flat to 8 cm, i.e. 2 cm OUTSIDE the grid it
  saw. That is the multi-base augmentation working -- for a policy that reads its joint state.
  The VLA fine-tune failed because SmolVLA is image-dominant and the image is base-invariant:
  for it the bases were indistinguishable and it averaged. Same data, different observation
  channel, opposite result. The probe would have shown that in a minute.
- Relative joint deltas from base 0 alone miss by < 1 cm at 8 cm. Predicting deltas instead of
  absolute targets buys most of the robustness for free. Worth a line in the docs: absolute
  joint targets are the worst action space for base robustness; delta joints are nearly as
  good as relative EE at these shifts (the arm's Jacobian barely changes over 8 cm).
- The feasibility row is the fair-test boundary: x+8 keeps 49% of frames, y+2 100%.

Caveats to keep honest:
- Probe policies see state only; the image channel is untested here. A real VLA needs an
  adapter (LeRobot policy -> (image, state) -> action). No checkpoint exists any more, and the
  user does not want one trained; the adapter is a stub until someone has a policy to test.
- 1.3 cm floor for the grid MLP at b=0 is fit error (3000 full-batch steps, 5.8k pairs).
- Grid bases at z=0 only; z shifts untested. Trivial to add (`--train-grid`, offsets list).
- The probe is on TRAINING frames (same demos). Held-out demos next (idea 5).

Next: ambiguity metric (idea 2) as `ambiguity.py` -- the miss a policy incurs if it cannot tell
the bases apart and averages their actions. Should reproduce the 3.3 cm found the hard way.

### 20:35 -- ambiguity metric ran. `scripts/experiment/ambiguity.py`, seconds.
Miss at the tool point if a policy averages the joint actions of a 3x3 base grid (cm, median / p90):

| grid half-width | absolute joints | joint deltas | +- 4 cm in z too (27 bases): absolute | deltas |
|---|---|---|---|---|
| 2 cm | 2.1 / 2.9 | 0.1 / 0.4 | 4.4 / 5.2 | 0.3 / 1.1 |
| 4 cm | 4.3 / 5.9 | 0.3 / 0.8 | 5.8 / 7.4 | 0.4 / 1.3 |
| 6 cm | 6.7 / 9.0 | 0.4 / 1.1 | 7.7 / 10.2 | 0.5 / 1.3 |

- Averaging absolute joint targets costs about the grid half-width in xy, and z spread is the
  worst offender (+-2 xy with +-4 z already 4.4 cm). can_multi varied heights 4-16 cm and lost
  3.3 cm: same mechanism, now predicted from the data in seconds instead of found after a
  fine-tune and 20 rollouts.
- Joint deltas keep it under 0.5 cm everywhere. Relative EE is 0 by construction.
- So the rule for multi-base augmentation: never with absolute joint targets unless the policy
  is state-conditioned enough to tell the bases apart; deltas or EE-relative, and it is safe.
- Feasible-from-all-bases frames fall fast (287 -> 70 at +-6): the augmentation also throws
  away data unless the plan is re-chunked per base. Worth its own number in the tool.

Cross-check against the simulator, from this afternoon's (now deleted) runs: the FIXED replay's
jaw-to-can distance at the closing row was 6.2 cm at y-6, 8.1 at y-8, 8.2 at x+8. The offline
probe's replay row says 6.0 / 8.0 / 8.0. Same number, no Isaac. That is the validation the
offline metric needed, at least for the reference it is exact for.

Open: the open-loop replay of a retargeted demo never lifted the can in sim, even with a dwell
at the grasp. Jaw axis at the closing row unmeasured (probe was cut off by the deletion). Parked:
the user wants no rollouts, and the offline metric does not need the replay to grasp.

### 21:00 -- user's idea: score the eval data by its likelihood under the policy (LLM-style)
Diffusion/flow policies are generative over action chunks, so "is this chunk something the
model would generate" is a density question. Flow matching = CNF, so exact:
    log p(x1 | obs) = log N(x0) - int_0^1 div v_theta(x_t, t | obs) dt      (x_t along the ODE)
Divergence by Hutchinson (E_eps eps^T J eps) with 2-4 probe vectors, ~10 Euler steps. Diffusion:
ELBO from the weighted denoising loss. Cheap proxy for both: the training loss on the chunk,
averaged over t. Model-agnostic proxy: distance of the true chunk to K sampled chunks.
Caveats: densities, so relative only (same model, same normalisation); OOD data can still get
high likelihood (Nalisnick 2019) -- fine for perturbations of in-distribution data, which is
what the base-shift probe feeds it; scores the whole chunk.
Why better than my point-prediction probe: multimodal policies average to nowhere; the
likelihood sees every mode. Also gives per-episode perplexity = outlier/curation tool, and a
way to say which of two datasets a held-out demo fits better.
Plan: tiny conditional FM policy on 5-D state -> action (exact div by autograd, no Hutchinson
needed at 5-D), base-0 vs grid, score = NLL(action_b | state_b). Then a LeRobot adapter that
uses `policy.forward()` loss + `sample_actions` for SmolVLA when a checkpoint exists.

### 21:10 -- why eval loss != success, and what it does to the probes
Loss is on the demonstrator's states (policy's own drift never scored); every frame and dim
weighted equally (success = ~5 grasp frames, 1-2 cm); likelihood rewards covering the human
variety, success rewards committing to one mode (sharper "overfit" checkpoints roll out
better); demonstrator noise is in the labels. So: offline scores only as CONTRASTS under a
controlled perturbation on the same frames (NLL(b) - NLL(0)), only at the grasp rows, only in
cm against the physical tolerance. They detect one failure mode; they do not predict success.
To do: `--grasp-window N` in offline_probe.py (rows within N of the jaw-closing row, using the
grip channel), report both all-frames and grasp-window numbers.

### 21:30 -- test bed for "does the offline score relate to success": PushT + lerobot/diffusion_pusht
Why PushT: the canonical diffusion-policy task, dataset + checkpoint on the hub, 2-D physics env
on CPU (seconds per episode), and gym-pusht can reset to a demo's initial state -- so each demo
episode gets (a) the policy's loss/NLL on its chunks, (b) contrast scores under perturbation,
(c) the policy's success from that same initial state. Then correlate across episodes. That is
the honest per-episode version of "loss vs success". Checkpoint-level (lowest val loss is not
the best checkpoint) would need a training run; later if the per-episode result is interesting.
Grasp-window numbers on the can probe: same picture as all-frames (replay = shift, grid MLP
flat ~1.4 cm, deltas < 1 cm), 59 grasp frames at base 0. So on this data the averaging did not
hide anything; keep both columns anyway.

### 22:10 -- PushT set-up notes (so nobody repeats them)
- lerobot 0.6.1 + `lerobot/diffusion_pusht` (old checkpoint): its normalisation buffers are
  REJECTED on load ("unexpected keys"); the processor pipeline must be rebuilt by hand. And the
  buffers are ImageNet mean/std for the image, not the dataset's stats. Built from dataset
  stats the policy scores 1/32; the loss looked sane (0.04) and gave no hint. Lesson: a sane
  loss value is not a check that the observation path is right.
- gym-pusht needs pymunk < 7 (`add_collision_handler` removed in 7).
- Original zarr (diffusion-policy site, pusht_cchi_v7_replay.zarr) has the 5-D state incl. the
  block pose; episode order matches lerobot/pusht (checked ep 0-2). gym-pusht
  `reset(options={"reset_to_state": state5})` works. Env render == dataset image at the same
  state (mean abs diff 4.7/255).
- Per-episode plan: 206 episodes, every 5th frame scored (loss x4 noise draws, 8 sampled chunks,
  state perturbed +20 px with the image fixed), one batched rollout per episode from its own
  start (38 policy calls x 0.86 s per batch of 32 envs).

### 22:40 -- flow-matching likelihood probe ran. `scripts/experiment/flow_probe.py`, ~2 min CPU.
Exact NLL (nats per 5-D action, normalisation Jacobian included) of the re-solved action given
the re-solved state, and of the base-0 action given the re-solved state; median over 4 directions:

| policy | score | 0 | 2 cm | 4 cm | 6 cm | 8 cm |
|---|---|---|---|---|---|---|
| flow, base 0 | NLL(action_b) | -10.4 | -9.4 | -7.5 | -5.4 | -2.5 |
| flow, base 0 | NLL(action_0) | -10.4 | -5.8 | 4.3 | 16.7 | 36.3 |
| flow, base 0 | miss of sampled mean | 1.1 | 1.3 | 1.7 | 2.2 | 2.8 |
| flow, grid   | NLL(action_b) | -9.2 | -9.2 | -9.2 | -9.0 | -8.7 |
| flow, grid   | NLL(action_0) | -9.2 | -6.1 | 1.8 | 11.1 | 22.7 |
| flow, grid   | miss of sampled mean | 1.4 | 1.4 | 1.4 | 1.5 | 1.5 |

- The contrast NLL(b) - NLL(0) on the correct action: base-0 policy +7.9 nats at 8 cm, grid
  policy +0.5. That is the memoriser / generaliser split as a likelihood, no rollout.
- The grid policy assigns the base-0 action 32 nats LESS at 8 cm: it knows the action must
  change with the base. Good sign for the metric: it is not just "everything gets less likely
  off-distribution".
- The base-0 flow's sampled mean misses only 2.8 cm at 8 cm (the MLP missed 4-10): a SiLU flow
  conditioned on state extrapolates a bit. Its likelihood still says clearly it is off; the
  point miss under-reports it. Which is the point of using the likelihood.
- Caveat: tiny 5-D state-only policies; the image channel is untested. The SmolVLA adapter is
  the same maths with Hutchinson for the divergence (50-D chunks) -- when there is a checkpoint.
- Training loss ~0.21-0.23 for both; 4000 steps, batch 512, Euler 40 steps for the ODE.

### 23:05 -- PushT result: offline scores on a demo do not predict success from its start
206 episodes x 2 rollouts of lerobot/diffusion_pusht from each demo's own initial state:
per-rollout success 60% (reported 65%); 90 episodes succeed twice, 68 once, 48 never.
AUC for "succeeds" (0.5 = nothing): loss 0.55, loss p90 0.55, min sample distance 0.55, mean
distance 0.55, sample spread 0.48, state reliance 0.57. Spearman vs best coverage: all within
+-0.10. Demo LENGTH -- how long the human took from that start -- has AUC 0.36 (rho -0.14):
the start's difficulty predicts the policy's success better than anything the model says about
the demo. 13 starts got coverage 0.0 in both rollouts; their losses are ordinary (0.003-0.012).
So: measured, not argued. The loss on a demonstration is about how well the model imitates
that human's path; success from that start is about the policy finding its own path. They are
different questions. Same for sample distance and the perturbation contrast at the demo's
states. Offline scores of this kind are instruments for data questions (does the model contain
this behaviour; is an augmentation safe; is the policy base-invariant), not success predictors.
Consequence for the library: say so in the docs, and never ship a "predicted success" number.

### 2026-09-11 22:30 -> 12 00:10 -- shipped 0.3.0 (commit 3664b4a)
What went in: `place`, `report`, `ambiguity`, `probe`, `export --action delta`, `serve`
(FastAPI + one plain page), Dockerfile, `omnibase/eval.py`, tests/test_eval.py (8 tests,
whole suite 46 green). Verified on the can data: place 30 s -> (0.18, 0.28) 52.5%, chunked
78.5%; ambiguity reproduces 2.1/4.3/6.7; probe reproduces replay = shift, kNN grid flat.
Design decisions worth remembering:
- sweep records `reach` (per-cell counts), `grasps`, `still`, `extent` per episode, so place
  and report are pure aggregations of the plans JSON. Old plans without `reach` are refused.
- probe's policy interface: `policy(state, where) -> action` radians, optional `.nll`.
  `where` = (episode, frame) so an adapter can fetch the image. kNN references built in;
  no torch in the package.
- service = subprocess per job, one worker, the page shows the exact argv. No auth/upload.
- UI: user said "don't make it look AI slop" -> monospace, no rounded cards, greyscale map,
  the command line under the form. Labels are the CLI flags.
- README says plainly: none of this predicts success (PushT).
Not done / next: a LeRobot adapter example for probe (needs a checkpoint); `place` with
the comfort score per cell (score_map is per episode and slow); probe on held-out demos;
the user's PushT-style validation of probe on a policy with images.
