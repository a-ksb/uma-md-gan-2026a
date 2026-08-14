#!/usr/bin/env python3
"""
09_fig_parity_by_state.py

Generate the manuscript figure "05b_parity_by_state.png": the UMA-vs-DFT
parity plot resolved into the two branches (bonding states of Ga_admol),
with per-branch linear fits, plus the structural-fingerprint histogram of
d(Ga_admol - nearest top-layer Ga) for each branch.

Branch definition: residual dE_UMA - dE_DFT smoothed with a 7-point
(14 fs) centered rolling mean, threshold 6 meV/atom.
  branch 0 (blue) : lifted          (low residual, no Ga-Ga contact)
  branch 1 (red)  : surface-engaged (high residual, weak Ga-Ga bonds)

Input : outputs/08_parity_cluster_merged.csv (from 08_cluster_analysis.py)
Output: outputs/05b_parity_by_state.pdf (vector, for the manuscript)
        outputs/05b_parity_by_state.png (preview)

The figure is drawn at the actual print size (single column of the
two-column wlscirep layout, ~3.4 in wide), so the fonts below appear
at their nominal point size, comparable to the 10 pt body text.

Usage
-----
    python3 09_fig_parity_by_state.py [outputs_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7})


def main(outdir: str) -> None:
    df = pd.read_csv(os.path.join(outdir, "08_parity_cluster_merged.csv"))
    df["resid_meV"] = df["residual_eV_per_atom"] * 1000
    r = df["resid_meV"].rolling(7, center=True, min_periods=1).mean()
    df["branch"] = (r > 6).astype(int)

    x = "dft_energy_relative_meV_per_atom"
    y = "uma_energy_relative_meV_per_atom"
    df[x] = df["dft_energy_relative_eV_per_atom"] * 1000
    df[y] = df["uma_energy_relative_eV_per_atom"] * 1000
    lo, hi = df[df.branch == 0], df[df.branch == 1]

    # vertical panels: drawn at actual print size (single column ~3.4 in)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 6.0))

    # ---- (a) parity by branch ----
    ax = axes[0]
    lim = [-8, 252]
    ax.plot(lim, lim, "--", color="gray", lw=1, label="$y=x$")
    stats = {}
    for sub, c, lab in [(lo, "tab:blue", "lifted"),
                        (hi, "tab:red", "surface-engaged")]:
        ax.scatter(sub[x], sub[y], s=7, alpha=0.65, color=c,
                   edgecolors="none", label=lab)
        m, b = np.polyfit(sub[x], sub[y], 1)
        xx = np.linspace(sub[x].min(), sub[x].max(), 10)
        ax.plot(xx, m * xx + b, color=c, lw=1.2)
        stats[lab] = (m, b)
        print(f"{lab}: slope={m:.3f} intercept={b:+.1f} meV/atom n={len(sub)}")
    k1 = "lifted"
    k2 = "surface-engaged"
    txt = (f"lifted:  $y={stats[k1][0]:.3f}x{stats[k1][1]:+.1f}$\n"
           f"engaged: $y={stats[k2][0]:.3f}x{stats[k2][1]:+.1f}$")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta E_\mathrm{DFT}$ (meV/atom)")
    ax.set_ylabel(r"$\Delta E_\mathrm{MLIP}$ (meV/atom)")
    ax.legend(loc="upper left")
    ax.set_title("(a)", loc="left")

    # ---- (b) structural fingerprint ----
    ax = axes[1]
    bins = np.linspace(2.2, 4.3, 32)
    ax.hist(lo["d_adGa_topGa_min"], bins=bins, alpha=0.6, color="tab:blue",
            label="lifted branch")
    ax.hist(hi["d_adGa_topGa_min"], bins=bins, alpha=0.6, color="tab:red",
            label="surface-engaged branch")
    ax.set_xlabel(r"$d$(Ga$_\mathrm{admol}$ – nearest top-layer Ga) (Å)")
    ax.set_ylabel("count")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # headroom so legend clears bars
    ax.legend()
    ax.set_title("(b)", loc="left")

    plt.tight_layout()
    # align panel (b) to the square panel (a): same left edge and width
    axes[0].apply_aspect()
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    axes[1].set_position([p0.x0, p1.y0, p0.width, p1.height])
    out_pdf = os.path.join(outdir, "05b_parity_by_state.pdf")
    plt.savefig(out_pdf)
    print("wrote", out_pdf)
    out_png = os.path.join(outdir, "05b_parity_by_state.png")
    plt.savefig(out_png, dpi=300)
    print("wrote", out_png)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    main(os.path.abspath(outdir))
