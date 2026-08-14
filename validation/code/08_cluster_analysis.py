#!/usr/bin/env python3
"""
08_cluster_analysis.py

Analyze the two parallel branches ("clusters") in the UMA-vs-CASTEP
parity plot by merging the per-frame residuals with the structural
descriptors computed by 07_compute_structural_descriptors.py.

Inputs (in outputs/)
--------------------
05_parity_table.csv
07_structural_descriptors.csv
07b_env_descriptors.csv
07c_bondcounts.csv

Outputs (in outputs/)
---------------------
08_parity_cluster_merged.csv : merged per-frame table (residual + descriptors)
08_cluster_analysis.png      : 3-panel diagnostic figure
(stdout)                     : band statistics, correlations, threshold
                               classification accuracies, per-state fits

Key finding (2026-07)
---------------------
Residual = dE_UMA - dE_DFT splits into two bands corresponding to the
bonding state of the admolecule Ga (adGa):
  low  band (~1-3 meV/atom): adGa upright, no Ga-Ga contact with the top
                             surface layer (d(adGa-topGa) >~ 3.5 A)
  high band (~10  meV/atom): adGa leaning into the surface, weak Ga-Ga
                             bonding (d(adGa-topGa) ~ 2.5-3.0 A), closer
                             to surface H
GaNH dissociation is NOT the cause (d(adGa-adN) ~ 2.0 A throughout) and
the identity of the N-bonded surface site is not decisive either.

Usage
-----
    python3 08_cluster_analysis.py [outputs_dir]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(outdir: str) -> None:
    p = pd.read_csv(os.path.join(outdir, "05_parity_table.csv"))
    s = pd.read_csv(os.path.join(outdir, "07_structural_descriptors.csv"))
    e = pd.read_csv(os.path.join(outdir, "07b_env_descriptors.csv"))
    b = pd.read_csv(os.path.join(outdir, "07c_bondcounts.csv"))
    df = p.merge(s, on="time_fs").merge(e, on="time_fs").merge(b, on="time_fs")
    df["resid_meV"] = df["residual_eV_per_atom"] * 1000
    df["tilt"] = df["adGa_z"] - df["adN_z"]
    df.to_csv(os.path.join(outdir, "08_parity_cluster_merged.csv"), index=False)

    # ---------- residual bands (smoothed threshold, hysteresis-free) ----------
    r_smooth = df["resid_meV"].rolling(7, center=True, min_periods=1).mean()
    df["band"] = (r_smooth > 6).astype(int)  # 1 = high-residual band
    lo, hi = df[df.band == 0], df[df.band == 1]
    print(f"low  band: n={len(lo)}  residual {lo.resid_meV.mean():.2f} "
          f"± {lo.resid_meV.std():.2f} meV/atom")
    print(f"high band: n={len(hi)}  residual {hi.resid_meV.mean():.2f} "
          f"± {hi.resid_meV.std():.2f} meV/atom")

    # ---------- descriptor ranking ----------
    cols = ["d_adGa_adN", "adGa_z", "adN_z", "tilt", "lift_dz",
            "d_adGa_H_min", "d_adN_H_min", "nH3_adGa", "nH4_adGa",
            "nGa30_adGa", "nGa33_adGa", "d_adGa_topGa_min", "nH4_adN",
            "nGaGa30", "nGaGa33", "nGaH19", "top_z_std", "top_z_max",
            "sumGaGa_inv", "d_adN_topGa_min"]
    print(f"\n{'descriptor':>18} {'corr':>6} {'thr-acc':>8} "
          f"{'low mean':>10} {'high mean':>10}")
    for c in cols:
        rr = np.corrcoef(df[c], df["resid_meV"])[0, 1]
        acc_best, thr_best = 0.0, None
        for thr in np.unique(df[c]):
            pred = (df[c] > thr).astype(int)
            acc = max((pred == df.band).mean(), ((1 - pred) == df.band).mean())
            if acc > acc_best:
                acc_best, thr_best = acc, thr
        print(f"{c:>18} {rr:+6.2f} {acc_best:8.2f} "
              f"{lo[c].mean():10.2f} {hi[c].mean():10.2f}   thr={thr_best:.2f}")

    # ---------- band segments in time ----------
    df["seg"] = (df.band.diff() != 0).cumsum()
    segs = df.groupby("seg").agg(
        t0=("time_fs", "min"), t1=("time_fs", "max"), n=("band", "size"),
        band=("band", "first"), resid=("resid_meV", "mean"),
        partner=("partner1", lambda x: x.mode()[0]),
        lift=("lift_idx", lambda x: x.mode()[0]),
        dGaGa=("d_adGa_topGa_min", "mean"))
    print("\nband segments:")
    print(segs[segs.n >= 5].to_string(float_format=lambda v: f"{v:7.2f}"))

    # ---------- per-state parity fits ----------
    x, y = df["dft_energy_relative_eV_per_atom"], df["uma_energy_relative_eV_per_atom"]
    for name, sub in [("low ", lo), ("high", hi)]:
        m, c0 = np.polyfit(sub["dft_energy_relative_eV_per_atom"],
                           sub["uma_energy_relative_eV_per_atom"], 1)
        print(f"parity fit {name}: slope={m:.3f} intercept={c0*1000:.1f} meV/atom")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    sc = ax.scatter(x, y, c=df["d_adGa_topGa_min"], cmap="coolwarm_r",
                    s=16, alpha=0.85, edgecolors="none")
    lim = [-0.005, max(x.max(), y.max()) + 0.01]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    plt.colorbar(sc, ax=ax, label="d(adGa – nearest top Ga) (Å)")
    ax.set_xlabel("ΔE_DFT (eV/atom)")
    ax.set_ylabel("ΔE_UMA (eV/atom)")
    ax.set_title("(a) Parity, colored by adGa–surface-Ga distance")

    ax = axes[1]
    sc = ax.scatter(df["d_adGa_topGa_min"], df["resid_meV"],
                    c=df["d_adGa_H_min"], cmap="viridis",
                    s=16, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="d(adGa – nearest surface H) (Å)")
    rr = np.corrcoef(df["d_adGa_topGa_min"], df["resid_meV"])[0, 1]
    ax.set_xlabel("d(adGa – nearest top-layer Ga) (Å)")
    ax.set_ylabel("residual (meV/atom)")
    ax.set_title(f"(b) Residual vs adGa coordination (r = {rr:+.2f})")

    ax = axes[2]
    ax.plot(df.time_fs / 1000, df.resid_meV, color="crimson", lw=1.2)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("residual (meV/atom)", color="crimson")
    ax2 = ax.twinx()
    for col, colr in [("d_adGa_topGa_min", "steelblue"), ("d_adGa_H_min", "seagreen")]:
        ax2.plot(df.time_fs / 1000,
                 df[col].rolling(7, center=True, min_periods=1).mean(),
                 color=colr, lw=1.2, label=col)
    ax2.set_ylabel("distance (Å)")
    ax2.legend(loc="upper center", fontsize=9)
    ax.set_title("(c) Residual and adGa environment vs time")

    plt.tight_layout()
    out = os.path.join(outdir, "08_cluster_analysis.png")
    plt.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    main(os.path.abspath(outdir))
