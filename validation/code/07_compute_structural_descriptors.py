#!/usr/bin/env python3
"""
07_compute_structural_descriptors.py

Compute per-frame structural descriptors of the GaNH admolecule and its
environment from the txt_snapshots (JSON) files, for the parity-cluster
analysis of the UMA vs CASTEP validation.

Inputs
------
outputs/txt_snapshots/snapshot_*.txt : JSON snapshots (751 frames)
    keys used: symbols, positions_A, cell_A, time_fs

Outputs (written next to txt_snapshots, i.e. into outputs/)
-----------------------------------------------------------
07_structural_descriptors.csv : admolecule geometry per frame
07b_env_descriptors.csv       : adGa/adN neighbor counts per frame
07c_bondcounts.csv            : global bond-type counts per frame

Atom-role identification (from frame 0):
  adGa   = Ga atom with maximum z  (admolecule Ga)
  adN    = highest N below the vacuum-wrapped bottom layer (admolecule N)
  nhH    = surface H closest to adN (the H of the NH unit)
  top_ga = 16 Ga atoms of the topmost surface layer (10.0 < z < 12.2 A)
  surf_h = surface H adatoms (11.5 < z < 14.5 A), excluding nhH

Usage
-----
    python3 07_compute_structural_descriptors.py [outputs_dir]
    (default outputs_dir: ./outputs relative to this script's parent)
"""
import csv
import glob
import json
import os
import sys

import numpy as np


def main(outdir: str) -> None:
    files = sorted(glob.glob(os.path.join(outdir, "txt_snapshots", "snapshot_*.txt")))
    if not files:
        sys.exit(f"no snapshots found under {outdir}/txt_snapshots")
    print(f"{len(files)} snapshot files")

    # ---------- identify atom roles from frame 0 ----------
    d0 = json.load(open(files[0]))
    sym = np.array(d0["symbols"])
    pos0 = np.array(d0["positions_A"])
    cell = np.array(d0["cell_A"])
    inv = np.linalg.inv(cell.T)
    z0 = pos0[:, 2]

    ga_idx = np.where(sym == "Ga")[0]
    n_idx = np.where(sym == "N")[0]
    h_idx = np.where(sym == "H")[0]

    adGa = int(ga_idx[np.argmax(z0[ga_idx])])
    adN = int(n_idx[np.argmax(np.where(z0[n_idx] < 20, z0[n_idx], -1))])
    top_ga = [int(i) for i in ga_idx if 10.0 < z0[i] < 12.2 and i != adGa]
    surf_h_all = [int(h) for h in h_idx if 11.5 < z0[h] < 14.5]
    sub_n = [int(i) for i in n_idx if 9.5 < z0[i] < 11.5]

    def mic(dv):
        """minimum-image convention displacement(s)"""
        f = dv @ inv.T
        f -= np.round(f)
        return f @ cell

    d_nh = [np.linalg.norm(mic(pos0[h] - pos0[adN])) for h in surf_h_all]
    nhH = surf_h_all[int(np.argmin(d_nh))]
    adatom_h = [h for h in surf_h_all if h != nhH]

    print(f"adGa={adGa} adN={adN} nhH={nhH} "
          f"top_ga={len(top_ga)} surface H adatoms={len(adatom_h)}")

    def pdists(pos, ii, jj):
        dv = pos[ii][:, None, :] - pos[jj][None, :, :]
        f = dv @ inv.T
        f -= np.round(f)
        dv = f @ cell
        return np.linalg.norm(dv, axis=2)

    rows_a, rows_b, rows_c = [], [], []
    top_ga_plus = np.array(top_ga + [adGa])

    for fpath in files:
        d = json.load(open(fpath))
        pos = np.array(d["positions_A"])
        t = d["time_fs"]

        def dist(i, j):
            return float(np.linalg.norm(mic(pos[i] - pos[j])))

        # ---- 07: admolecule geometry ----
        top_z = np.array([pos[i, 2] for i in top_ga])
        med = float(np.median(top_z))
        lift_i = top_ga[int(np.argmax(top_z))]
        dN = sorted((dist(adN, i), i) for i in top_ga)
        partners = [i for dd, i in dN if dd < 2.3]
        dH = sorted((dist(adGa, h), h) for h in adatom_h)
        rows_a.append(dict(
            time_fs=t,
            d_adGa_adN=dist(adGa, adN),
            d_adN_nhH=dist(adN, nhH),
            adGa_z=float(pos[adGa, 2]) - med,
            adN_z=float(pos[adN, 2]) - med,
            lift_dz=float(np.max(top_z)) - med,
            lift_idx=lift_i,
            n_partners=len(partners),
            partner1=partners[0] if partners else -1,
            partner2=partners[1] if len(partners) > 1 else -1,
            d_adGa_H_min=dH[0][0],
            h_near=dH[0][1],
            d_adN_topGa_min=dN[0][0],
        ))

        # ---- 07b: adGa/adN neighbor counts ----
        dHs = sorted(dist(adGa, h) for h in adatom_h)
        dGas = sorted(dist(adGa, g) for g in top_ga)
        dNHs = sorted(dist(adN, h) for h in adatom_h)
        rows_b.append(dict(
            time_fs=t,
            nH3_adGa=sum(1 for x in dHs if x < 3.0),
            nH4_adGa=sum(1 for x in dHs if x < 4.0),
            nGa30_adGa=sum(1 for x in dGas if x < 3.0),
            nGa33_adGa=sum(1 for x in dGas if x < 3.3),
            d_adGa_topGa_min=dGas[0],
            d_adN_H_min=dNHs[0],
            nH4_adN=sum(1 for x in dNHs if x < 4.0),
        ))

        # ---- 07c: global bond counts (top layer + adspecies) ----
        dgg = pdists(pos, top_ga_plus, top_ga_plus)
        np.fill_diagonal(dgg, 99)
        dgh = pdists(pos, top_ga_plus, np.array(surf_h_all))
        dnh_ = pdists(pos, np.array([adN]), np.array(surf_h_all))
        dgn_top = pdists(pos, top_ga_plus, np.array([adN]))
        dgn_sub = pdists(pos, top_ga_plus, np.array(sub_n))
        zt = pos[top_ga_plus[:-1], 2]
        rows_c.append(dict(
            time_fs=t,
            nGaGa30=int((dgg < 3.0).sum() // 2),
            nGaGa33=int((dgg < 3.3).sum() // 2),
            nGaH19=int((dgh < 1.9).sum()),
            nNH12=int((dnh_ < 1.2).sum()),
            nGaN22_ad=int((dgn_top < 2.2).sum()),
            minGaN_sub=float(dgn_sub.min()),
            nGaN_sub_22=int((dgn_sub < 2.2).sum()),
            top_z_std=float(np.std(zt)),
            top_z_max=float(np.max(zt) - np.median(zt)),
            sumGaGa_inv=float((1.0 / dgg[dgg < 3.5]).sum() / 2),
        ))

    for name, rows in [("07_structural_descriptors.csv", rows_a),
                       ("07b_env_descriptors.csv", rows_b),
                       ("07c_bondcounts.csv", rows_c)]:
        out = os.path.join(outdir, name)
        with open(out, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    main(os.path.abspath(outdir))
