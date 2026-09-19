"""Figure: effect of refinement strategy on validator-defined validity, diversity
and residual failure modes (supervised vs RL vs HITL generators).

All numbers are recomputed from the per-sample validator output
(``samples.jsonl``) of each evaluation run (N = 1,000 grammar-constrained
samples per model, temperature 1.0, top-p 0.9) and cross-checked against the
run's ``eval_report.json``.

The supervised model is A1 (``gen_ablate_A1``): both the RL
(``gen_rl_stage1``) and HITL (``gen_hitl_round1``) generators were fine-tuned
from that checkpoint (``resume_ckpt`` in their configs), so A1 is the matched
parent.

Usage:
    python figures/fig_refinement_validity.py [--runs PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE.parent / "runs" / "ablate"

MODELS = [  # (label, eval run dir)
    ("Supervised", "eval_ablate_A1"),
    ("RL", "eval_ablate_A5_with_rl"),
    ("HITL", "eval_ablate_A6_with_hitl"),
]
# validator reason -> display name; every reason the validator can emit
REASONS = [
    ("preference", "Site preference"),
    ("same_group", "Same group"),
    ("symmetry", "Symmetry"),
    ("single_x", "Single-X"),
    ("rule_invalid", "Grammar"),
]

# validated categorical slots (dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def load(runs: Path) -> list[dict]:
    rows = []
    for label, d in MODELS:
        samples = [json.loads(l) for l in open(runs / d / "samples.jsonl")]
        valid = [s for s in samples if s["valid"]]
        reasons = collections.Counter(s["reason"] for s in samples if not s["valid"])
        unique_valid = len({s["layers"] for s in valid})

        rep = json.load(open(runs / d / "eval_report.json"))["generator"]
        assert abs(rep["validity_rate"] - len(valid) / len(samples)) < 1e-9, d
        assert rep["unique_mxenes_count"] == unique_valid, d
        assert dict(reasons) == rep["invalid_reasons"], d

        lo, hi = wilson(len(valid), len(samples))
        rows.append(dict(model=label, run=d, ckpt=rep["ckpt"], n=len(samples),
                         n_valid=len(valid), validity=len(valid) / len(samples),
                         ci_lo=lo, ci_hi=hi, unique_valid=unique_valid,
                         n_invalid=len(samples) - len(valid),
                         **{f"rej_{k}": reasons.get(k, 0) for k, _ in REASONS}))
    return rows


def style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, width=0.6, length=3)
    ax.set_axisbelow(True)


def panel_a(ax, rows):
    xs = [100 * r["validity"] for r in rows]
    ys = [r["unique_valid"] for r in rows]
    for r, x, y in zip(rows, xs, ys):
        ax.errorbar(x, y, xerr=[[x - 100 * r["ci_lo"]], [100 * r["ci_hi"] - x]],
                    fmt="o", ms=7, color=INK, mfc=INK, mec="white", mew=1.2,
                    ecolor=MUTED, elinewidth=1.2, capsize=3, zorder=3)
    offsets = {"Supervised": (0, 9, "center"), "RL": (0, 9, "center"),
               "HITL": (10, -4, "left")}
    for r, x, y in zip(rows, xs, ys):
        dx, dy, ha = offsets[r["model"]]
        ax.annotate(f"{r['model']}\n{x:.1f}% · {y} unique", (x, y),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="bottom" if dy > 0 else "top", fontsize=7.5, color=INK,
                    linespacing=1.25)
    ax.set_xlim(94.0, 99.2)
    ax.set_ylim(140, 172)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("Validator-defined validity (% of 1,000 samples)")
    ax.set_ylabel("Unique valid layer-resolved candidates")
    ax.text(1.0, 1.02, "Error bars: 95% Wilson interval", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=INK_2)
    style_axis(ax)


def panel_b(ax, rows):
    shown = [(k, name) for k, name in REASONS if any(r[f"rej_{k}"] for r in rows)]
    absent = [name for k, name in REASONS if (k, name) not in shown]
    n_types = len(shown)
    width = 0.8 / n_types
    gap = 0.02  # surface gap between adjacent bars
    for j, (k, name) in enumerate(shown):
        for i, r in enumerate(rows):
            x = i + (j - (n_types - 1) / 2) * width
            v = r[f"rej_{k}"]
            ax.bar(x, v, width=width - gap, color=SERIES[j], linewidth=0,
                   label=name if i == 0 else None, zorder=2)
            ax.text(x, v + 0.8, str(v), ha="center", va="bottom", fontsize=7,
                    color=INK if v else MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['model']}\n({r['n_invalid']} rejected)" for r in rows])
    ax.set_ylabel("Rejected samples (of 1,000)")
    ax.set_ylim(0, max(r[f"rej_{k}"] for r in rows for k, _ in shown) * 1.22)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(axis="x", length=0)
    leg = ax.legend(frameon=False, fontsize=7, loc="lower left",
                    bbox_to_anchor=(0.0, 1.0), ncol=n_types, columnspacing=1.2,
                    handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    if absent:
        ax.text(0.5, 0.97, "No rejections: " + ", ".join(absent),
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                color=INK_2)
    style_axis(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    rows = load(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "fig_refinement_validity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.labelcolor": INK,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.05]))
    panel_a(ax_a, rows)
    panel_b(ax_b, rows)
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.16, 1.06, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=INK, va="bottom")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(args.out / "fig_refinement_validity.png", dpi=300,
                facecolor="white")
    for r in rows:
        print(f"{r['model']:<11} valid {r['n_valid']}/{r['n']} "
              f"({100*r['validity']:.1f}%, 95% CI {100*r['ci_lo']:.1f}–{100*r['ci_hi']:.1f}) "
              f"unique {r['unique_valid']}  rejected "
              + ", ".join(f"{k}={r[f'rej_{k}']}" for k, _ in REASONS))


if __name__ == "__main__":
    main()
