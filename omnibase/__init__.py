"""OmniBase -- make free-flown gripper demonstrations reachable by putting the robot somewhere else.

Nothing in a hand-held recording says where the robot was standing, because there was no robot.
So the base pose is not a property of the hardware, it is a free variable of the data: choose
it, and choose it again whenever it stops working.

    import omnibase as ob

    chain = ob.so101()                                   # or build your own Chain
    _, hands = ob.from_parquet("datasets/mine", episode=6)
    pos, quat, _ = hands[0]

    cells = ob.base_grid(span=0.42, step=0.02, height=0.08)
    F = ob.feasibility(chain, pos, quat, cells)
    chunks, held = ob.chunk([F], cells, window=21)

    print(f"{held.mean():.0%} of frames held across {len(chunks)} chunk(s)")
    print("against %.0f%% for the single best base" % (100 * ob.best_fixed(F, cells)[1]))

numpy and scipy only. No simulator.
"""
from .chain import Chain, Joint
from .data import from_parquet, normalise_quats
from .plan import (SCORE_TERMS, Chunk, ascii_map, base_grid, best_fixed, best_spot, chunk,
                   feasibility, score_map, yield_curve)
from .robots import describe, load, so101

__version__ = "0.1.0"
__all__ = [
    "Chain", "Joint", "Chunk",
    "base_grid", "best_fixed", "chunk", "feasibility", "yield_curve",
    "score_map", "best_spot", "ascii_map", "SCORE_TERMS",
    "so101", "load", "describe",
    "from_parquet", "normalise_quats",
]
