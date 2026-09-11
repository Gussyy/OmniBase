"""Base-shift results -> table + plot. Reads $B/results.jsonl (one line per replayed episode)."""
import json, collections, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/data/baseshift")
rows = [json.loads(l) for l in (B / "results.jsonl").read_text().splitlines() if l.strip()]
base_rows = {r["episode"]: r["rows"] for r in rows if r["root"].endswith("can_0")}
tab = collections.OrderedDict()
for r in rows:
    tab.setdefault(r["tag"], {}).setdefault("fixed" if r["root"].endswith("/can_0") else "retargeted", []).append(r)
def cm(tag): return 0 if tag == "0" else int(tag[1:])
order = sorted(tab, key=lambda t: (cm(t), t))
lines = ["| shift | fixed rows: lifted/n | jaw-to-can at close (cm) | re-solved: retained frames | lifted/n | jaw-to-can (cm) |", "|---|---|---|---|---|---|"]
curves = collections.defaultdict(dict)
for t in order:
    f = tab[t].get("fixed", []); g = tab[t].get("retargeted", []) if t != "0" else f
    # only chunks that close the jaws on the way in count; one demo chunk starts already closed, 20 cm up
    kept = sum(x["rows"] for x in g); tot = sum(base_rows.values()) or 1
    f = [x for x in f if x.get("closed_at")]; g = [x for x in g if x.get("closed_at")]
    def lifted(v): return sum(x["lifted"] for x in v)
    def dcl(v):
        d = [x["d_close"] for x in v if x.get("d_close") is not None]; return 100 * np.median(d) if d else float("nan")
    lines.append(f"| {t} | {lifted(f)}/{len(f)} | {dcl(f):.1f} | {100*kept/tot:.0f}% ({len(g)} eps) | {lifted(g)}/{len(g)} | {dcl(g):.1f} |")
    for ax in ("x", "y"):
        if t == "0" or t[0] == ax:
            d = cm(t) * (1 if t == "0" or t[1] == "+" else -1); curves[ax][d] = (dcl(f), dcl(g), lifted(f) / max(len(f), 1), lifted(g) / max(len(g), 1))
md = "\n".join(lines); print(md); (B / "table.md").write_text(md + "\n", encoding="utf-8")
fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex="col")
for col, ax in enumerate(("x", "y")):
    ds = sorted(curves[ax])
    axs[0][col].plot(ds, [curves[ax][d][0] for d in ds], "o-", label="fixed rows (replay as trained)")
    axs[0][col].plot(ds, [curves[ax][d][1] for d in ds], "s--", label="re-solved from shifted base"); axs[0][col].set_title(f"base shift along {ax}"); axs[0][col].grid(alpha=.3)
    axs[1][col].plot(ds, [100 * curves[ax][d][2] for d in ds], "o-"); axs[1][col].plot(ds, [100 * curves[ax][d][3] for d in ds], "s--"); axs[1][col].set_xlabel("shift (cm)"); axs[1][col].grid(alpha=.3)
axs[0][0].set_ylabel("jaw-to-can at close (cm)"); axs[1][0].set_ylabel("lifted (%)"); axs[0][0].legend(fontsize=8); fig.tight_layout()
fig.savefig(B / "baseshift.png", dpi=130); print(f"wrote {B/'table.md'}, {B/'baseshift.png'}")
