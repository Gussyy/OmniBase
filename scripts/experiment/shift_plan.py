"""Move every chunk's base by (dx, dy, dz): the same demonstrations, re-solved from a shifted stand.

    python shift_plan.py can_base2.json 0.04 0 0 out.json
"""
import json, sys
from pathlib import Path
src, dx, dy, dz, dst = sys.argv[1], *map(float, sys.argv[2:5]), sys.argv[5]
p = json.loads(Path(src).read_text(encoding="utf-8"))
for r in p["episodes"]:
    for c in r.get("chunks", []):
        c["bases"] = [[round(b[0] + dx, 4), round(b[1] + dy, 4), round(b[2] + dz, 4)] for b in c["bases"]]
p["cfg"]["home"] = None   # the sweep's home is not where these are solved from any more
Path(dst).write_text(json.dumps(p, indent=1), encoding="utf-8")
print(f"{sum(len(r.get('chunks', [])) for r in p['episodes'])} chunks shifted by ({dx}, {dy}, {dz}) -> {dst}")
