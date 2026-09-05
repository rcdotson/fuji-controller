#!/usr/bin/env python3
"""Map real-world subject distance -> GF lens focus position counts.

Reads the Saleae SPI captures listed in data/dataset_overview.tsv, decodes them
with fuji_spi, and extracts the focus position the body settled on for each
measured subject distance. Then fits candidate distance->counts models and
plots them.

Focus position sources (see protocol/README.md):
  - body->lens  cmd 0x15 tag2 = absolute focus motor TARGET position
  - lens->body  cmd 0x08 tag1 = focus position FEEDBACK
Both are signed BE16 and agree once a move settles, so the last AF target of a
capture is taken as the focus setting for that station.

Usage:
  python analysis/focus_distance.py [--lens Lens1] [--camera Camera1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOFTWARE = HERE.parent
sys.path.insert(0, str(SOFTWARE))

import fuji_spi  # noqa: E402

DATA = SOFTWARE / "data"
YD_TO_M = 0.9144

# Both lenses are GF250mm F4 R LM OIS WR (same model, different serials); the
# ident block in the gfx100sii_init.txt capture spells it out. Because the model
# is shared, the counts-per-mm scale -- and therefore the fitted b -- must agree
# between the two datasets; that is the cross-check in compare_plot().
# The focal length is used only to turn count residuals into a depth-of-field
# tolerance, never by the fit itself.
LENS_MODEL = "GF250mm F4"
FOCAL_MM = 250.0
APERTURE_N = 4.0     # wide open — the tightest tolerance, so a conservative band
COC_MM = 0.04        # circle of confusion for the 44x33mm GFX sensor

# Reference data-viz palette, categorical slots 1-3 (validated all-pairs).
C_DATA = "#2a78d6"   # measured points
C_INV = "#eb6834"    # inverse-distance fit
C_QUAD = "#1baf7a"   # quadratic-in-1/d fit
C_MUTED = "#8a8a85"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def focus_events(path: Path) -> tuple[list, list]:
    """Return (targets, feedback), each a list of (time_s, counts)."""
    _, pkts = fuji_spi.load(path, fuji_spi.DEFAULT_GAP_S)
    targets, feedback = [], []
    for p in pkts:
        if len(p.raw) != 4 or p.is_idle or p.is_ack:
            continue
        if p.direction == "tx" and p.cmd_base == 0x15 and p.tag2 == 2:
            targets.append((p.t, p.payload_s16))
        elif p.direction == "rx" and p.cmd_base == 0x08 and p.tag2 == 1:
            if p.payload_s16 != 32767:  # encoder-range sentinel, not a position
                feedback.append((p.t, p.payload_s16))
    return targets, feedback


def load_overview(path: Path) -> list[dict]:
    """dataset_overview.tsv has a stray extra tab in its header, so split on
    tabs and drop empty cells rather than trusting DictReader's alignment."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header = [c.strip() for c in lines[0].split("\t") if c.strip()]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("\t") if c.strip()]
        if len(cells) != len(header):
            raise SystemExit(f"unexpected column count in {path}: {ln!r}")
        rows.append(dict(zip(header, cells)))
    return rows


def collect(lens: str, camera: str) -> list[dict]:
    recs = []
    for r in load_overview(DATA / "dataset_overview.tsv"):
        if r["Lens"] != lens or r["Camera"] != camera:
            continue
        targets, feedback = focus_events(DATA / r["Filename"])
        recs.append({
            "file": r["Filename"],
            "station": r["Station"],
            "dist_yd": float(r["Distance_yds"]),
            "dist_m": float(r["Distance_yds"]) * YD_TO_M,
            "target": targets[-1][1] if targets else None,
            "feedback": feedback[-1][1] if feedback else None,
            "n_af": len(targets),
            "targets": targets,
            "fb_tail": [v for _, v in feedback[-6:]],
        })
    return recs


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def fit_report(name: str, basis: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of y on the given design matrix; returns coefs + stats."""
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    pred = basis @ coef
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    dof = len(y) - basis.shape[1]
    return {
        "name": name,
        "coef": coef,
        "resid": resid,
        "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
        "rmse": float(np.sqrt(ss_res / len(y))),
        "rmse_adj": float(np.sqrt(ss_res / dof)) if dof > 0 else float("nan"),
        "dof": dof,
    }


def build_fits(d_m: np.ndarray, counts: np.ndarray) -> dict[str, dict]:
    inv = 1.0 / d_m
    ones = np.ones_like(d_m)
    return {
        "inv": fit_report("counts = a + b/d", np.column_stack([ones, inv]), counts),
        "inv2": fit_report("counts = a + b/d + c/d^2",
                           np.column_stack([ones, inv, inv ** 2]), counts),
        "poly2": fit_report("counts = a + b*d + c*d^2",
                            np.column_stack([ones, d_m, d_m ** 2]), counts),
    }


def dof_tolerance_counts(b: float) -> tuple[float, float]:
    """How many counts of focus error the depth of field permits.

    The inverse model counts = a + b/d is, physically, counts linear in focus
    extension: x_mm = f^2/(1000*d_m), so counts - a = (1000*b/f^2) * x_mm.
    Defocus at the image plane is tolerable up to +/- N*c, so that same scale
    converts the DoF into counts.
    """
    counts_per_mm = 1000.0 * b / FOCAL_MM ** 2
    return counts_per_mm, counts_per_mm * APERTURE_N * COC_MM


def eval_fit(key: str, coef: np.ndarray, d_m: np.ndarray) -> np.ndarray:
    if key == "inv":
        return coef[0] + coef[1] / d_m
    if key == "inv2":
        return coef[0] + coef[1] / d_m + coef[2] / d_m ** 2
    return coef[0] + coef[1] * d_m + coef[2] * d_m ** 2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(recs: list[dict], fits: dict, out: Path, lens: str, camera: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d_yd = np.array([r["dist_yd"] for r in recs])
    d_m = np.array([r["dist_m"] for r in recs])
    counts = np.array([r["counts"] for r in recs], dtype=float)

    grid_yd = np.linspace(d_yd.min() * 0.85, d_yd.max() * 1.12, 400)
    grid_m = grid_yd * YD_TO_M

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    def style(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(alpha=0.25, lw=0.8, color=C_MUTED)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_MUTED)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=C_TEXT2, labelsize=9)

    a, b = fits["inv"]["coef"]
    _, tol = dof_tolerance_counts(b)

    # --- panel 1: counts vs distance -------------------------------------
    inv_curve = eval_fit("inv", fits["inv"]["coef"], grid_m)
    ax1.fill_between(grid_yd, inv_curve - tol, inv_curve + tol, color=C_INV,
                     alpha=0.13, lw=0,
                     label=f"depth of field at f/{APERTURE_N:g}  (±{tol:.0f} counts)")
    ax1.plot(grid_yd, inv_curve, lw=2, color=C_INV,
             label=f"a + b/d   R²={fits['inv']['r2']:.3f}")
    ax1.plot(grid_yd, eval_fit("inv2", fits["inv2"]["coef"], grid_m), lw=2,
             color=C_QUAD, ls=(0, (5, 3)),
             label=f"a + b/d + c/d²   R²={fits['inv2']['r2']:.3f}")
    ax1.plot(d_yd, counts, "o", ms=9, color=C_DATA, mec="#fcfcfb", mew=2,
             ls="none", label="measured AF target", zorder=5)
    # Push each label away from the fit line (sign of its residual) so it never
    # sits on the curve; 117 yd has two captures at the same x, so stagger those.
    seen: dict[float, int] = {}
    for r, res in zip(recs, fits["inv"]["resid"]):
        n = seen.get(r["dist_yd"], 0)
        seen[r["dist_yd"]] = n + 1
        # leftmost point would run off the axis if labelled to its left
        right = bool(n) or r["dist_yd"] == d_yd.min()
        ha, dx = ("left", 11) if right else ("right", -11)
        dy, va = (11, "bottom") if res >= 0 else (-11, "top")
        ax1.annotate(f"{r['file'].removesuffix('.txt')}  {r['counts']:+d}",
                     (r["dist_yd"], r["counts"]), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, va=va, fontsize=8, color=C_TEXT2)
    ax1.set_xscale("log")
    ax1.set_xticks([50, 75, 100, 150, 250, 400, 600])
    ax1.set_xticks([], minor=True)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlim(grid_yd[0], grid_yd[-1] * 1.05)
    ax1.set_xlabel("subject distance (yards, log scale)", color=C_TEXT2, fontsize=10)
    ax1.set_ylabel("focus position (counts)", color=C_TEXT2, fontsize=10)
    ax1.set_title(f"{lens} ({LENS_MODEL}) / {camera} — focus counts vs distance",
                  color=C_TEXT, fontsize=12, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, labelcolor=C_TEXT2, loc="upper right")
    style(ax1)

    # --- panel 2: linearised on 1/d --------------------------------------
    x_grid = 1000.0 / grid_m
    ax2.fill_between(x_grid, inv_curve - tol, inv_curve + tol, color=C_INV,
                     alpha=0.13, lw=0)
    ax2.plot(x_grid, inv_curve, lw=2, color=C_INV)
    ax2.plot(1000.0 / d_m, counts, "o", ms=9, color=C_DATA,
             mec="#fcfcfb", mew=2, ls="none", zorder=5)
    # Selective direct labels: the two endpoints anchor the axis, the rest are
    # already named in panel 1.
    for r in (min(recs, key=lambda r: r["dist_yd"]),
              max(recs, key=lambda r: r["dist_yd"])):
        ax2.annotate(f"{r['dist_yd']:.0f} yd", (1000 / r["dist_m"], r["counts"]),
                     textcoords="offset points", xytext=(10, -2), ha="left",
                     va="top", fontsize=8, color=C_TEXT2)
    ax2.set_xlim(0, (1000.0 / d_m).max() * 1.18)
    ax2.set_xlabel("reciprocal distance  1000/d  (m⁻¹ ×10³)", color=C_TEXT2, fontsize=10)
    ax2.set_ylabel("focus position (counts)", color=C_TEXT2, fontsize=10)
    ax2.set_title("Linearised — counts is straight in 1/d",
                  color=C_TEXT, fontsize=12, loc="left", pad=12)
    style(ax2)

    # --- panel 3: residuals ----------------------------------------------
    width = 0.36
    idx = np.arange(len(recs))
    ax3.axhspan(-tol, tol, color=C_INV, alpha=0.13, lw=0,
                label=f"within DoF at f/{APERTURE_N:g}")
    ax3.axhline(0, color=C_MUTED, lw=1)
    ax3.bar(idx - width / 2, fits["inv"]["resid"], width, color=C_INV,
            label=f"a + b/d  (RMSE {fits['inv']['rmse']:.0f})")
    ax3.bar(idx + width / 2, fits["inv2"]["resid"], width, color=C_QUAD,
            label=f"a + b/d + c/d²  (RMSE {fits['inv2']['rmse']:.0f})")
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"{r['dist_yd']:.0f} yd\n{r['file'].removesuffix('.txt')}"
                         for r in recs], fontsize=8)
    ax3.set_ylim(-tol * 1.65, tol * 1.65)
    ax3.set_ylabel("residual (counts)", color=C_TEXT2, fontsize=10)
    ax3.set_title("Fit residuals vs focus tolerance",
                  color=C_TEXT, fontsize=12, loc="left", pad=12)
    ax3.legend(frameon=False, fontsize=9, labelcolor=C_TEXT2, loc="lower left")
    style(ax3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"\nwrote {out}")


def compare_plot(sets: list[dict], out: Path) -> None:
    """Overlay two body/lens datasets.

    Both lenses are the same model, so the fitted b (counts per unit 1/d) must
    agree between them. It doesn't, and the third panel shows why: the AF search
    excursion says how far the focus motor wandered before settling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.patch.set_facecolor("#fcfcfb")

    def style(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(alpha=0.25, lw=0.8, color=C_MUTED)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_MUTED)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=C_TEXT2, labelsize=9)

    all_yd = np.concatenate([[r["dist_yd"] for r in s["recs"]] for s in sets])
    grid_yd = np.linspace(all_yd.min() * 0.85, all_yd.max() * 1.12, 400)
    grid_m = grid_yd * YD_TO_M

    for s in sets:
        recs, color = s["recs"], s["color"]
        d_yd = np.array([r["dist_yd"] for r in recs])
        d_m = np.array([r["dist_m"] for r in recs])
        counts = np.array([r["counts"] for r in recs], dtype=float)
        a, b = s["fits"]["inv"]["coef"]

        ax1.plot(grid_yd, a + b / grid_m, lw=2, color=color)
        ax1.plot(d_yd, counts, "o", ms=9, color=color, mec="#fcfcfb", mew=2,
                 ls="none", zorder=5, label=s["label"])
        ax2.plot(1000.0 / grid_m, a + b / grid_m, lw=2, color=color,
                 label=f"{s['label']}   b={b:,.0f}  R²={s['fits']['inv']['r2']:.3f}")
        ax2.plot(1000.0 / d_m, counts, "o", ms=9, color=color, mec="#fcfcfb",
                 mew=2, ls="none", zorder=5)

    ax1.set_xscale("log")
    ax1.set_xticks([50, 75, 100, 150, 250, 400, 600])
    ax1.set_xticks([], minor=True)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlim(grid_yd[0], grid_yd[-1] * 1.05)
    ax1.set_xlabel("subject distance (yards, log scale)", color=C_TEXT2, fontsize=10)
    ax1.set_ylabel("focus position (counts)", color=C_TEXT2, fontsize=10)
    ax1.set_title(f"{LENS_MODEL} on two bodies — focus counts vs distance",
                  color=C_TEXT, fontsize=12, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, labelcolor=C_TEXT2, loc="upper right")
    style(ax1)

    ax2.set_xlim(0, (1000.0 / (all_yd.min() * YD_TO_M)) * 1.1)
    ax2.set_xlabel("reciprocal distance  1000/d  (m⁻¹ ×10³)", color=C_TEXT2, fontsize=10)
    ax2.set_ylabel("focus position (counts)", color=C_TEXT2, fontsize=10)
    ax2.set_title("Same lens model ⇒ slopes should match", color=C_TEXT,
                  fontsize=12, loc="left", pad=12)
    ax2.legend(frameon=False, fontsize=9, labelcolor=C_TEXT2, loc="upper left")
    style(ax2)

    # --- panel 3: how far the AF hunted before settling -------------------
    labels, spans, colors = [], [], []
    for s in sets:
        if labels:  # blank slot separating the two bodies
            labels.append("")
            spans.append(0)
            colors.append("none")
        for r in s["recs"]:
            vals = [v for _, v in r["targets"]] or [r["counts"]]
            labels.append(f"{r['file'].removesuffix('.txt').removesuffix('_maybebad')}"
                          f"  ({r['dist_yd']:.0f} yd)")
            spans.append(max(vals) - min(vals))
            colors.append(s["color"])
    idx = np.arange(len(labels))
    ax3.bar(idx, spans, 0.68, color=colors)
    ax3.set_xticks(idx)
    ax3.set_xticklabels(labels, fontsize=8, rotation=45, ha="right",
                        rotation_mode="anchor")
    ax3.set_ylabel("AF search excursion (counts)", color=C_TEXT2, fontsize=10)
    ax3.set_title("How far the AF hunted before settling", color=C_TEXT,
                  fontsize=12, loc="left", pad=12)
    style(ax3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


# ---------------------------------------------------------------------------

def analyse(lens: str, camera: str, csv_out: Path | None) -> tuple[list[dict], dict]:
    recs = collect(lens, camera)
    if not recs:
        raise SystemExit(f"no captures for {lens}/{camera}")

    print(f"===== {lens} / {camera} =====")
    print(f"{'file':20} {'station':10} {'yd':>5} {'m':>7} {'AF moves':>8} "
          f"{'target':>7} {'feedback':>9}")
    for r in recs:
        r["counts"] = r["target"] if r["target"] is not None else r["feedback"]
        tgt = "-" if r["target"] is None else str(r["target"])
        fb = "-" if r["feedback"] is None else str(r["feedback"])
        print(f"{r['file']:20} {r['station']:10} {r['dist_yd']:5.0f} "
              f"{r['dist_m']:7.1f} {r['n_af']:8d} {tgt:>7} {fb:>9}")

    print("\nAF target sequence per capture (settling behaviour):")
    for r in recs:
        seq = ", ".join(f"{v:+d}@{t:.2f}s" for t, v in r["targets"]) or "(no AF drive)"
        vals = [v for _, v in r["targets"]]
        span = f"  [search span {max(vals) - min(vals)}]" if vals else ""
        print(f"  {r['file']:20} {seq}{span}")

    d_m = np.array([r["dist_m"] for r in recs])
    counts = np.array([r["counts"] for r in recs], dtype=float)
    fits = build_fits(d_m, counts)

    print("\nfits (d in metres, counts = focus position):")
    for key in ("inv", "inv2", "poly2"):
        f = fits[key]
        coefs = "  ".join(f"{c:+.6g}" for c in f["coef"])
        print(f"  {f['name']:26} coef=[{coefs}]  R²={f['r2']:.4f}  "
              f"RMSE={f['rmse']:.1f} counts  (dof={f['dof']})")

    a, b = fits["inv"]["coef"]
    counts_per_mm, tol = dof_tolerance_counts(b)
    print("\n  inverse model, ready to use:")
    print(f"    counts(d_m) = {a:.2f} + {b:.1f}/d_m")
    print(f"    d_m(counts) = {b:.1f} / (counts - ({a:.2f}))")
    print(f"    infinity asymptote: {a:.1f} counts")
    print(f"    implied scale: {counts_per_mm:.0f} counts per mm of focus extension")
    print(f"    depth of field at f/{APERTURE_N:g}, coc {COC_MM} mm: "
          f"±{tol:.0f} counts")
    worst = np.abs(fits["inv"]["resid"]).max()
    print(f"    worst residual: {worst:.0f} counts "
          f"({'within' if worst <= tol else 'OUTSIDE'} that tolerance)")

    print("\n  what the AF actually resolved (distance implied by each count):")
    for r, res in zip(recs, fits["inv"]["resid"]):
        implied = b / (r["counts"] - a)
        print(f"    {r['file']:20} {r['dist_yd']:5.0f} yd -> AF focused as if "
              f"{implied / YD_TO_M:6.0f} yd   (residual {res:+6.1f} counts)")

    # Repeat stations are the only direct measure of AF repeatability.
    by_station: dict[str, list[int]] = {}
    for r in recs:
        by_station.setdefault(r["station"], []).append(r["counts"])
    repeats = {k: v for k, v in by_station.items() if len(v) > 1}
    if repeats:
        print("\n  AF repeatability (repeat captures at one station):")
        for st, vals in repeats.items():
            print(f"    {st:10} {vals} -> spread {max(vals) - min(vals)} counts")

    if csv_out:
        with csv_out.open("w", newline="") as f:
            f.write("file,station,distance_yd,distance_m,af_moves,af_target,"
                    "feedback,counts_used\n")
            for r in recs:
                f.write(f"{r['file']},{r['station']},{r['dist_yd']:.0f},"
                        f"{r['dist_m']:.2f},{r['n_af']},{r['target']},"
                        f"{r['feedback']},{r['counts']}\n")
        print(f"\nwrote {csv_out}")
    return recs, fits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", default="Lens1")
    ap.add_argument("--camera", default="Camera1")
    ap.add_argument("--plot", type=Path, default=HERE / "focus_vs_distance.png")
    ap.add_argument("--csv", type=Path, default=HERE / "focus_vs_distance.csv")
    ap.add_argument("--compare", action="store_true",
                    help="analyse Lens1/Camera1 and Lens2/Camera2 and overlay them")
    args = ap.parse_args()

    if not args.compare:
        recs, fits = analyse(args.lens, args.camera, args.csv)
        plot(recs, fits, args.plot, args.lens, args.camera)
        return

    sets = []
    for lens, camera, color, label in (
            ("Lens1", "Camera1", C_DATA, "Lens1 / Camera1 (GFX100S II)"),
            ("Lens2", "Camera2", C_INV, "Lens2 / Camera2 (GFX50S II)")):
        suffix = f"{lens}_{camera}".lower()
        recs, fits = analyse(lens, camera, HERE / f"focus_vs_distance_{suffix}.csv")
        plot(recs, fits, HERE / f"focus_vs_distance_{suffix}.png", lens, camera)
        sets.append({"recs": recs, "fits": fits, "color": color, "label": label})
        print()

    b1 = sets[0]["fits"]["inv"]["coef"][1]
    b2 = sets[1]["fits"]["inv"]["coef"][1]
    print("===== cross-check =====")
    print(f"Both lenses are {LENS_MODEL}, so the fitted b must agree.")
    print(f"  Lens1/Camera1 b = {b1:9,.0f} counts·m")
    print(f"  Lens2/Camera2 b = {b2:9,.0f} counts·m   ({(b2 / b1 - 1) * 100:+.0f}%)")

    # Hold the scale at the Camera1 value and let only the offset float: if the
    # Camera2 points followed a single 1/d law, this would fit them well.
    d2 = np.array([r["dist_m"] for r in sets[1]["recs"]])
    y2 = np.array([r["counts"] for r in sets[1]["recs"]], dtype=float)
    a_fixed = float(np.mean(y2 - b1 / d2))
    resid = y2 - (a_fixed + b1 / d2)
    print(f"\n  Lens2/Camera2 refit with b held at {b1:,.0f} (offset only):")
    print(f"    a = {a_fixed:.1f},  RMSE = {np.sqrt((resid ** 2).mean()):.0f} counts")
    for r, res in zip(sets[1]["recs"], resid):
        print(f"    {r['file']:20} {r['dist_yd']:5.0f} yd  residual {res:+7.1f}")

    compare_plot(sets, HERE / "focus_vs_distance_compare.png")


if __name__ == "__main__":
    main()
