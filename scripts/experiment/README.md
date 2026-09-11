# Experiment scripts

Evidence, not the package. These produced the numbers in `docs/experiment/`; the shipped
commands (`omnibase place / report / ambiguity / probe`) are the parts that survived.

| script | what it showed |
|---|---|
| `offline_probe.py` | the base-shift probe with kNN and MLP toy policies: replay = the shift exactly, one-base policy memorises, grid policy flat |
| `flow_probe.py` | the same probe as an exact likelihood under a tiny flow-matching policy (+8 nats at 8 cm for one base, +0.5 for a grid) |
| `ambiguity.py` | the first version of `omnibase ambiguity` |
| `pusht_probe.py` | PushT + `lerobot/diffusion_pusht`: no offline score of a demonstration predicts success from its start (AUC 0.48-0.57) |
| `pipeline_baseshift.sh`, `shift_plan.py`, `baseshift_table.py` | the simulator replay of re-solved demonstrations under a base shift (Isaac Sim; the open-loop replay never grasped and was parked) |

`pusht_probe.py` needs `lerobot`, `gym-pusht`, `pymunk<7`, `zarr` and the original PushT zarr
from the diffusion-policy site; see the notebook for the set-up traps.
