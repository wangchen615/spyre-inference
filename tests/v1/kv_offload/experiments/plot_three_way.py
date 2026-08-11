#!/usr/bin/env python3
"""Plot the three-way (recompute / host-reload / HBM-reuse) sweep.

Two figures:

  1. latency vs context length -- one curve per path. Log-log, because the x-range
     spans 64x and recompute is superlinear while the transfer paths are linear;
     on linear axes the four shortest lengths collapse into the origin.

  2. dual y-axis -- speedup ratios on the left (rec/reload and reload/reuse),
     blocks reloaded on the right. The right axis is the *evidence* axis: it shows
     that every point moved a full L/128-block prefix, so the left-axis ratios
     compare like with like and no "reload" is a device hit in disguise.

Runs anywhere matplotlib is installed -- it does NOT need Spyre, the venv, or the
pod. The measured sweep is embedded below so the figures are reproducible from
this file alone; pass --csv to plot a different run.

    python3 scripts/plot_three_way.py                     # -> figures/ (png + pdf)
    python3 scripts/plot_three_way.py --csv mine.csv      # a later sweep
    python3 scripts/plot_three_way.py --outdir /tmp/figs --format png

CSV columns (header required, extra columns ignored):
    length,recompute_s,reload_s,reuse_s,loaded

To regenerate that CSV from a sweep's own PROGRESS file:
    python3 scripts/plot_three_way.py --from-progress results/full-sweep-*/PROGRESS \
        --csv-out sweep.csv

Caveat carried into every figure caption: reload/reuse is a LOWER bound, and
unevenly so. The `reuse` step spuriously reloads one decode block, which inflates
the reuse baseline -- 12.5% of reuse traffic at 1024 but only 0.2% at 65536. The
*direction* of the reload/reuse decay is therefore safe; its slope is not exact.
See docs/TRAFFIC_DESIGN.md and docs/LIMITATIONS.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The measured sweep. micro-g3.3-8b-instruct-1b, one AIU, block_size=128,
# pool_mult=1.5, host_gb=64, OMP_NUM_THREADS=8.
#
# Two sources, same host and same configuration:
#   * 1024 .. 65536  full-sweep-20260810-195315, finished 2026-08-11T00:53Z,
#                    verbatim from reference-results/full-sweep/PROGRESS.
#                    Cycles: 10 at 1024/4096/8192, 6 at 16384/32768/65536.
#   * 2048           traffic-2048, run 2026-08-11 to even out the x spacing --
#                    1024->4096 is a 4x gap where every other step is 2x, which
#                    stretches the leftmost segment of a log2 x-axis. Same driver,
#                    same env block, 10 cycles, loaded 16/16.
#
# Added later and separately, so it is worth being explicit that this is a
# cross-run splice: every ratio below is still computed within a single process,
# but 2048 vs its neighbours is a cross-process comparison. It interpolates
# cleanly on all three paths (12.147 / 0.743 / 0.589 against 5.846..24.995,
# 0.432..1.358, 0.352..1.045), which is the check that it belongs on the curve.
# ---------------------------------------------------------------------------
EMBEDDED_CSV = """\
length,cycles,recompute_s,reload_s,reuse_s,loaded
1024,10,5.846,0.432,0.352,8
2048,10,12.147,0.743,0.589,16
4096,10,24.995,1.358,1.045,32
8192,10,55.169,2.361,1.991,64
16384,6,119.755,4.657,3.912,128
32768,6,274.012,9.496,8.150,256
65536,6,714.463,19.435,17.201,512
"""

BLOCK_SIZE = 128

# Colour-blind-safe (Okabe-Ito). Recompute is the expensive path, so it gets the
# warning colour; the two transfer paths get cool tones.
C_RECOMPUTE = "#D55E00"  # vermillion
C_RELOAD = "#0072B2"  # blue
C_REUSE = "#009E73"  # green
C_BLOCKS = "#7F7F7F"  # grey -- the evidence curve, deliberately quiet


class Row:
    """One context length's measurements."""

    __slots__ = ("length", "cycles", "recompute", "reload", "reuse", "loaded")

    def __init__(self, length, cycles, recompute, reload_, reuse, loaded):
        self.length = int(length)
        self.cycles = int(cycles) if cycles not in (None, "") else 0
        self.recompute = float(recompute)
        self.reload = float(reload_)
        self.reuse = float(reuse)
        self.loaded = float(loaded)

    @property
    def expected_blocks(self) -> int:
        return self.length // BLOCK_SIZE

    @property
    def rec_over_reload(self) -> float:
        return self.recompute / self.reload

    @property
    def reload_over_reuse(self) -> float:
        return self.reload / self.reuse

    @property
    def full_reload(self) -> bool:
        return abs(self.loaded - self.expected_blocks) < 0.5


def parse_csv(text: str) -> list[Row]:
    reader = csv.DictReader(io.StringIO(text))
    missing = {"length", "recompute_s", "reload_s", "reuse_s", "loaded"} - set(
        reader.fieldnames or []
    )
    if missing:
        raise SystemExit(f"CSV is missing required column(s): {sorted(missing)}")
    rows = [
        Row(
            r["length"],
            r.get("cycles"),
            r["recompute_s"],
            r["reload_s"],
            r["reuse_s"],
            r["loaded"],
        )
        for r in reader
        if r.get("length", "").strip()
    ]
    if not rows:
        raise SystemExit("CSV contained no data rows")
    return sorted(rows, key=lambda r: r.length)


# PROGRESS table line, e.g.
#     1024     10        ok       292       5.846       0.432       0.352  13.53x  1.23x  8.0
_PROGRESS_RE = re.compile(
    r"^\s*(?P<length>\d+)\s+(?P<cycles>\d+)\s+(?P<status>\S+)\s+(?P<wall>\S+)\s+"
    r"(?P<rec>[\d.]+)\s+(?P<rel>[\d.]+)\s+(?P<reu>[\d.]+)\s+"
    r"[\d.]+x\s+[\d.]+x\s+(?P<loaded>[\d.]+)\s*$"
)


def parse_progress(path: Path) -> list[Row]:
    """Read a sweep driver's own PROGRESS file. Skips rows that did not succeed."""
    rows, skipped = [], []
    for line in path.read_text().splitlines():
        m = _PROGRESS_RE.match(line)
        if not m:
            continue
        if m.group("status") != "ok":
            skipped.append(f"{m.group('length')} ({m.group('status')})")
            continue
        rows.append(
            Row(
                m.group("length"),
                m.group("cycles"),
                m.group("rec"),
                m.group("rel"),
                m.group("reu"),
                m.group("loaded"),
            )
        )
    if skipped:
        print(f"  note: skipped non-ok rows: {', '.join(skipped)}", file=sys.stderr)
    if not rows:
        raise SystemExit(f"no completed rows parsed from {path}")
    return sorted(rows, key=lambda r: r.length)


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["length", "cycles", "recompute_s", "reload_s", "reuse_s", "loaded"])
        for r in rows:
            w.writerow(
                [r.length, r.cycles, r.recompute, r.reload, r.reuse, f"{r.loaded:g}"]
            )
    print(f"  wrote {path}")


def check_geometry(rows: list[Row]) -> None:
    """Warn loudly if any point is not a genuine full reload.

    A row where loaded < L/128 reloaded only part of its prefix, so the rest was a
    device hit mislabelled as a reload -- its reload_s is not comparable to the
    others and its reload/reuse is flattering. Plot it if you must, but know it.
    """
    bad = [r for r in rows if not r.full_reload]
    if not bad:
        print(f"  geometry OK: loaded == L/{BLOCK_SIZE} on all {len(rows)} rows")
        return
    print("  WARNING: partial reload -- these points are NOT comparable:", file=sys.stderr)
    for r in bad:
        pct = 100.0 * r.loaded / r.expected_blocks if r.expected_blocks else 0.0
        print(
            f"    length {r.length}: loaded {r.loaded:g} of {r.expected_blocks}"
            f" ({pct:.0f}%)",
            file=sys.stderr,
        )


def _pow2_ticks(ax, values, axis="x"):
    """Label the powers-of-two we actually measured, not matplotlib's decades."""
    labels = [f"{v:g}" for v in values]
    if axis == "x":
        ax.set_xticks(values)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
    else:
        ax.set_yticks(values)
        ax.set_yticklabels(labels)


def figure_latency(rows, plt, np, outdir, formats, dpi):
    """Figure 1: latency vs context length, one curve per path."""
    lengths = np.array([r.length for r in rows], dtype=float)
    rec = np.array([r.recompute for r in rows])
    rel = np.array([r.reload for r in rows])
    reu = np.array([r.reuse for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for y, colour, marker, label in (
        (rec, C_RECOMPUTE, "o", "recompute (evicted, no offload)"),
        (rel, C_RELOAD, "s", "host reload (offload hit, DRAM)"),
        (reu, C_REUSE, "^", "HBM reuse (device hit)"),
    ):
        ax.plot(
            lengths, y, marker=marker, color=colour, label=label,
            linewidth=1.9, markersize=6.5, markeredgecolor="white", markeredgewidth=0.7,
        )

    # Fitted slopes, not idealised ones. On log-log a power law is a straight line
    # whose slope IS the exponent, so a least-squares fit in log space reports what
    # the hardware actually did. Quoting an idealised "L^2" guide would be worse
    # than useless here: recompute's measured exponent is ~1.14, so an L^2 guide
    # visually accuses the data of a curvature it does not have.
    #
    # Do not confuse this 1.14 with phase 1's L^1.18 -- that fit is over wall-clock
    # runtime of a whole sweep, a different quantity that happens to land nearby.
    #
    # These are averages over the measured range, NOT asymptotic exponents. The local
    # exponent log(t2/t1)/log(L2/L1) between consecutive points is rising for
    # recompute (1.06 1.04 1.14 1.12 1.19 1.38) and converging to linear for the copy
    # (reload 0.78 0.87 0.80 0.98 1.03 1.03) -- consistent with a linear overhead term
    # plus a quadratic attention term that only dominates late, and with fixed
    # per-call cost amortising out of the copy. So a single slope UNDERSTATES
    # recompute past 65536: do not extrapolate L^1.14 beyond this range. The
    # mechanism claim (superlinear compute vs near-linear movement) is unaffected --
    # it is stronger locally (1.38 vs 1.03) than in the global fit (1.14 vs 0.91).
    #
    # Those are 6 gaps between 7 points. If a length is added or removed, recompute
    # them rather than editing a number: the lists are per-gap, not per-point.
    exps = {}
    guide_x = np.array([lengths[0], lengths[-1]])
    for y, colour, key in ((rec, C_RECOMPUTE, "recompute"),
                           (rel, C_RELOAD, "reload"),
                           (reu, C_REUSE, "reuse")):
        slope, intercept = np.polyfit(np.log(lengths), np.log(y), 1)
        exps[key] = slope
        ax.plot(guide_x, np.exp(intercept) * guide_x ** slope,
                ls=":", color=colour, lw=1.0, alpha=0.55, zorder=0)

    # The superlinear-vs-linear contrast is the mechanism, so state it as measured.
    ax.annotate(
        f"recompute $\\propto L^{{{exps['recompute']:.2f}}}$  (superlinear)",
        xy=(lengths[-1], rec[-1]), xytext=(-6, -22), textcoords="offset points",
        ha="right", fontsize=8.8, color=C_RECOMPUTE,
    )
    # ~L^0.9 rather than exactly L^1: per-block cost falls slightly with size as
    # fixed dispatch overhead amortises. "Sub-linear" is the honest word -- and it
    # only strengthens the argument, since it is further from recompute's 1.14.
    ax.annotate(
        f"transfer $\\propto L^{{{exps['reload']:.2f}}}$  (≈linear)",
        xy=(lengths[1], reu[1]), xytext=(6, -20), textcoords="offset points",
        ha="left", fontsize=8.8, color=C_RELOAD,
    )

    # Annotate the narrowest and widest recompute/reload gaps -- the two numbers a
    # reader wants -- as vertical spans placed between the curves they measure.
    for idx in (0, len(rows) - 1):
        r = rows[idx]
        mid = (r.recompute * r.reload) ** 0.5
        ax.annotate(
            "", xy=(r.length, r.recompute), xytext=(r.length, r.reload),
            arrowprops=dict(arrowstyle="<->", color=C_RECOMPUTE,
                            lw=1.0, alpha=0.6, shrinkA=3, shrinkB=3),
        )
        ax.annotate(
            f"{r.rec_over_reload:.1f}×", xy=(r.length, mid),
            xytext=(6 if idx == 0 else -6, 0), textcoords="offset points",
            ha="left" if idx == 0 else "right", va="center",
            fontsize=9.5, color=C_RECOMPUTE, fontweight="bold",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _pow2_ticks(ax, lengths, "x")
    ax.spines["right"].set_visible(False)
    # Headroom so the 13.5x / 36.8x callouts sit inside the axes rather than
    # spilling into the margin.
    ax.set_xlim(lengths[0] / 1.55, lengths[-1] * 1.55)
    ax.set_ylim(reu[0] / 2.2, rec[-1] * 2.6)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("prefill latency (s, median over cycles)")
    ax.set_title(
        "Cost of an evicted prefix: recompute vs host reload vs HBM reuse",
        fontsize=12, pad=10,
    )
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.3)
    ax.grid(True, which="minor", axis="y", ls=":", lw=0.4, alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    fig.text(
        0.5, 0.005,
        "Log-log. One engine per length; the three paths are produced by request "
        "order, not reconfiguration.\nEvery point reloaded its full L/128-block "
        "prefix. reload/reuse is a lower bound (see LIMITATIONS.md §4).",
        ha="center", va="bottom", fontsize=7.6, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return _save(fig, outdir, "latency_vs_length", formats, dpi, plt)


def figure_speedup_blocks(rows, plt, np, outdir, formats, dpi):
    """Figure 2: speedups on the left axis, blocks reloaded on the right."""
    lengths = np.array([r.length for r in rows], dtype=float)
    rec_rel = np.array([r.rec_over_reload for r in rows])
    rel_reu = np.array([r.reload_over_reuse for r in rows])
    loaded = np.array([r.loaded for r in rows])
    expected = np.array([r.expected_blocks for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    l1, = ax.plot(
        lengths, rec_rel, marker="o", color=C_RECOMPUTE,
        label="recompute / reload  (offload's win)",
        linewidth=2.0, markersize=6.5, markeredgecolor="white", markeredgewidth=0.7,
    )
    l2, = ax.plot(
        lengths, rel_reu, marker="s", color=C_RELOAD,
        label="reload / reuse  (DRAM round-trip penalty)",
        linewidth=2.0, markersize=6.5, markeredgecolor="white", markeredgewidth=0.7,
    )
    ax.axhline(1.0, color="0.7", lw=0.9, ls="--", zorder=0)
    # Caption placement is fiddlier than it looks: the right edge collides with the
    # 65536 reload/reuse label (which sits just above 1x by then) and the left edge
    # collides with the grey control marker at 1024, which starts at 8 blocks. Park
    # it under mid-span, where both series are well clear of the line.
    ax.annotate(
        "1× (no difference)", xy=(lengths[len(lengths) // 2], 1.0), xytext=(0, -14),
        textcoords="offset points", ha="center", fontsize=8, color="0.45",
    )

    for x, y in zip(lengths, rec_rel):
        ax.annotate(f"{y:.1f}×", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color=C_RECOMPUTE, fontweight="bold")
    # All reload/reuse labels go ABOVE the marker, never alternating. An earlier
    # version alternated above/below to avoid crowding, which was a mistake worth
    # recording: with 1.23 placed high and 1.30 placed low, the eye traces the
    # LABELS rather than the line and reads a dip between them. Label placement
    # must not imply a shape the data does not have, in either direction. Room is
    # bought with ylim headroom below instead.
    #
    # The series is NOT monotonic: it rises 1.23 -> 1.26 -> 1.30 to a peak at 4096,
    # then decays to 1.13. Adding 2048 made that head clearly visible where three
    # points had hidden it. The decay from the peak is the load-bearing claim
    # (transfer is not the bottleneck) and it is intact; the run-up is the short-
    # length regime where the spurious decode block is a large share of reuse
    # traffic (12.5% at 1024, 6.3% at 2048, 3.1% at 4096) and therefore where the
    # lower-bound bias is worst. Do not describe this line as monotonic.
    for x, y in zip(lengths, rel_reu):
        ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color=C_RELOAD, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _pow2_ticks(ax, lengths, "x")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("speedup ratio (×, log scale)", color="0.15")
    # The two series differ by ~25x, so leave real room below reload/reuse instead
    # of pinning the axis at 1.0 -- otherwise its labels collide with the 1x line
    # and the flat trend reads as if it were sitting on the floor.
    ax.set_ylim(min(rel_reu) / 2.6, max(rec_rel) * 2.4)
    ax.set_xlim(lengths[0] / 1.7, lengths[-1] * 1.7)
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.3)

    # Right axis: the evidence. Grey and dashed on purpose -- it is a control, not
    # a result, and it should not compete with the ratios for attention.
    ax2 = ax.twinx()
    l3, = ax2.plot(
        lengths, loaded, marker="D", color=C_BLOCKS, ls="--",
        label=f"blocks reloaded from host (= L/{BLOCK_SIZE}, full prefix)",
        linewidth=1.6, markersize=5.5, markerfacecolor="white",
        markeredgecolor=C_BLOCKS, markeredgewidth=1.3, zorder=1,
    )
    ax2.set_yscale("log", base=2)
    _pow2_ticks(ax2, loaded, "y")
    # Give the control curve generous headroom so it stays in the lower band and
    # does not cross the recompute/reload series or its callouts. The right axis
    # carries no comparison of its own -- only "is this L/128?" -- so compressing
    # it costs nothing and buys a readable left axis.
    ax2.set_ylim(min(loaded) / 2.0, max(loaded) * 12.0)
    ax2.set_ylabel(
        "KV blocks moved host→device (2.10 MB each)", color=C_BLOCKS
    )
    ax2.tick_params(axis="y", colors=C_BLOCKS)
    ax2.spines["right"].set_color(C_BLOCKS)
    ax2.spines["top"].set_visible(False)

    # Mark any point whose reload was only partial -- it invalidates that ratio.
    handles = [l1, l2, l3]
    bad = [(r.length, r.loaded) for r in rows if not r.full_reload]
    if bad:
        bx, by = zip(*bad)
        handles.append(
            ax2.scatter(bx, by, s=170, facecolors="none", edgecolors="red",
                        linewidths=2.0, zorder=5,
                        label="PARTIAL reload — ratio not comparable")
        )
        # The label above claims full reload; it is false once any point is partial.
        l3.set_label(f"blocks reloaded from host (want L/{BLOCK_SIZE})")

    ax.legend(handles, [h.get_label() for h in handles],
              loc="upper left", frameon=False, fontsize=9.2)

    ax.set_title(
        "Offload's advantage widens with length; the DRAM penalty does not",
        fontsize=12, pad=10,
    )

    # Describe the decay from its PEAK, not from the first point. The series rises
    # before it falls (peak at 4096), so "decays 1.23->1.13" would misdescribe the
    # very line it captions. Derived from the data so it stays true if points move.
    pk = max(range(len(rel_reu)), key=lambda i: rel_reu[i])
    peak_note = (
        f"decays {rel_reu[pk]:.2f}→{rel_reu[-1]:.2f}× from its peak at "
        f"{int(lengths[pk])}"
        if pk else f"decays {rel_reu[0]:.2f}→{rel_reu[-1]:.2f}×"
    )
    trend = (
        f"recompute/reload climbs {rec_rel[0]:.1f}→{rec_rel[-1]:.1f}× "
        f"(recompute is superlinear, a block copy is linear), while reload/reuse "
        f"{peak_note}:\ntransfer is not the "
        f"bottleneck. Grey axis is the control — all "
        f"{sum(1 for r in rows if r.full_reload)}/{len(rows)} points moved a full "
        f"prefix, so no reload is a device hit in disguise.\nreload/reuse is a lower "
        f"bound and unevenly so (spurious decode block: 12.5% of reuse traffic at "
        f"1024, 0.2% at 65536) — trust the direction, not the slope."
    )
    fig.text(0.5, 0.005, trend, ha="center", va="bottom", fontsize=7.6, color="0.35")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return _save(fig, outdir, "speedup_and_blocks", formats, dpi, plt)


def _save(fig, outdir, stem, formats, dpi, plt):
    written = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", format=fmt)
        written.append(path)
        print(f"  wrote {path}")
    plt.close(fig)
    return written


def _default_outdir() -> Path:
    """Where figures land when --outdir is not given.

    This file is vendored into two trees with different shapes:

      hillock-vmem   experiments/kvc-offload-evict/scripts/plot_three_way.py
      spyre-inference tests/v1/kv_offload/experiments/plot_three_way.py

    In the first, figures belong in the sibling `figures/` one level up. In the
    second the script sits directly in the experiments dir, so one level up is
    `tests/v1/kv_offload/` -- the wrong place, above the experiment entirely.
    Keying off the parent directory's NAME rather than a fixed number of levels
    keeps both correct, and keeps hillock's existing paths byte-identical.
    """
    here = Path(__file__).resolve().parent
    return (here.parent / "figures") if here.name == "scripts" else (here / "figures")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path,
                    help="CSV with length,recompute_s,reload_s,reuse_s,loaded "
                         "(default: the embedded measured sweep)")
    ap.add_argument("--from-progress", type=Path,
                    help="parse a sweep driver's PROGRESS file instead of a CSV")
    ap.add_argument("--csv-out", type=Path,
                    help="also write the parsed rows out as CSV")
    ap.add_argument("--outdir", type=Path, default=_default_outdir(),
                    help="where to write figures (default: the repo's figures/ "
                         "dir if this file sits in scripts/, else ./figures)")
    ap.add_argument("--format", default="png,pdf",
                    help="comma-separated output formats (default: png,pdf)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--show", action="store_true",
                    help="also open an interactive window (needs a display)")
    args = ap.parse_args()

    if args.csv and args.from_progress:
        ap.error("--csv and --from-progress are mutually exclusive")

    try:
        import matplotlib
    except ImportError:
        print(
            "matplotlib is not installed in this interpreter.\n"
            "This script is deliberately independent of the Spyre venv (which has\n"
            "no matplotlib and should not be perturbed on the pod). Either:\n"
            "  python3 -m pip install --user matplotlib   # or a throwaway venv\n"
            "or copy this file plus the CSV to a workstation and run it there.",
            file=sys.stderr,
        )
        return 2
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    if args.from_progress:
        rows = parse_progress(args.from_progress)
        src = str(args.from_progress)
    elif args.csv:
        rows = parse_csv(args.csv.read_text())
        src = str(args.csv)
    else:
        rows = parse_csv(EMBEDDED_CSV)
        src = "embedded measured sweep (full-sweep-20260810-195315)"

    print(f"data source: {src}")
    print(f"  {len(rows)} lengths: {', '.join(str(r.length) for r in rows)}")
    check_geometry(rows)
    if args.csv_out:
        write_csv(rows, args.csv_out)

    if len(rows) < 2:
        raise SystemExit("need at least 2 lengths to plot a trend")

    args.outdir.mkdir(parents=True, exist_ok=True)
    figure_latency(rows, plt, np, args.outdir, args.format.split(","), args.dpi)
    figure_speedup_blocks(rows, plt, np, args.outdir, args.format.split(","), args.dpi)

    # Print the same table the figures encode, so a text-only reader loses nothing.
    print("\n  length  recompute     reload      reuse   rec/reload  reload/reuse  loaded")
    for r in rows:
        flag = "" if r.full_reload else f"  <-- PARTIAL (want {r.expected_blocks})"
        print(f"  {r.length:6d}  {r.recompute:9.3f}  {r.reload:9.3f}  {r.reuse:9.3f}"
              f"  {r.rec_over_reload:9.2f}x  {r.reload_over_reuse:10.2f}x"
              f"  {r.loaded:6.0f}{flag}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
