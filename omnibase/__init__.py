"""OmniBase -- make free-flown gripper demonstrations reachable by putting the robot somewhere else.

Nothing in a hand-held recording says where the robot was standing, because there was no robot.
So the base pose is not a property of the hardware, it is a free variable of the data: choose
it, and choose it again whenever it stops working.

    import omnibase as ob

    chain = ob.so101()                                   # or build your own Chain
    ep = ob.load("datasets/mine", episode=6)          # LeRobot v2.1 or v3.0
    pos, quat = ep.hands[0].pos, ep.hands[0].quat

    cells = ob.base_grid(span=0.42, step=0.02, height=0.08)
    F = ob.feasibility(chain, pos, quat, cells)
    chunks, held = ob.chunk([F], cells, window=21)

    print(f"{held.mean():.0%} of frames held across {len(chunks)} chunk(s)")
    print("against %.0f%% for the single best base" % (100 * ob.best_fixed(F, cells)[1]))

numpy and scipy only. No simulator.
"""
from .chain import Chain, Joint
from .data import Episode, Hand, describe_dataset, from_parquet, load, normalise_quats
from .plan import (SCORE_TERMS, Chunk, Mount, arm_bases, ascii_map, base_grid,
                   best_fixed, best_spot, chunk, feasibility, home_index,
                   home_share, mount_feasibility, solve,
                   mount_grid, pair, score_map, yield_curve)
from .robots import describe, so101
from .robots import load as load_robot   # ob.load is the DATASET loader

__version__ = "0.1.0"
__all__ = [
    "Chain", "Joint", "Chunk",
    "base_grid", "best_fixed", "chunk", "feasibility", "solve", "yield_curve",
    "score_map", "best_spot", "ascii_map", "SCORE_TERMS",
    "Mount", "pair", "mount_grid", "mount_feasibility", "arm_bases",
    "home_index", "home_share",
    "so101", "load_robot", "describe",
    "load", "describe_dataset", "Episode", "Hand",
    "from_parquet", "normalise_quats",
]
