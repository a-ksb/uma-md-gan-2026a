#!/usr/bin/env python3
"""
Extract the Ga_admol (id 184)-N_admol (id 103) distance time series from a set of
UMA-MD dump files and write a manuscript figure (PDF) showing the dissociation and
re-formation of the GaNH unit.

- Sampling: every SAMPLE_EVERY steps (default 100 steps = 10 fs)
- Distances use the minimum image convention for a triclinic cell
- Output: ganh_distance.csv, ganh_distance.pdf (manuscript size, single-column width)
- Dissociated intervals (from manual trajectory inspection) are lightly shaded
"""
import glob
import itertools
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7})

GA_ID, N_ID = 184, 103
SAMPLE_EVERY = 100          # steps (0.1 fs/step -> 10 fs)
PS_PER_STEP = 1e-4

# Dissociated intervals [ps] identified by manual trajectory inspection
DISSOC_INTERVALS = [(23.40, 43.88), (100.51, 104.56),
                    (145.24, 145.67), (148.95, 150.0)]


def find_dump_files(directory):
    files = glob.glob(os.path.join(directory, '*.dump'))
    files.sort(key=lambda p: int(re.match(r'^(\d+)', os.path.basename(p)).group(1)))
    return files


def box_to_cell(l1, l2, l3):
    a1, a2, a3 = ([float(x) for x in l.split()] for l in (l1, l2, l3))
    xlo_b, xhi_b, xy = a1
    ylo_b, yhi_b, xz = a2
    zlo, zhi, yz = a3
    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    return np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    times, dists = [], []
    cell = inv = None
    for path in find_dump_files(here):
        print('reading', os.path.basename(path))
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith('ITEM: TIMESTEP'):
                    continue
                ts = int(f.readline())
                f.readline()
                natoms = int(f.readline())
                f.readline()
                b1, b2, b3 = f.readline(), f.readline(), f.readline()
                cols = f.readline().split()[2:]
                if ts % SAMPLE_EVERY or (times and ts * PS_PER_STEP <= times[-1]):
                    for _ in itertools.islice(f, natoms):
                        pass
                    continue
                if cell is None:
                    cell = box_to_cell(b1, b2, b3)
                    inv = np.linalg.inv(cell)
                ic = {c: i for i, c in enumerate(cols)}
                p = {}
                for _ in range(natoms):
                    q = f.readline().split()
                    aid = int(q[ic['id']])
                    if aid in (GA_ID, N_ID):
                        p[aid] = np.array([float(q[ic['x']]),
                                           float(q[ic['y']]),
                                           float(q[ic['z']])])
                frac = (p[GA_ID] - p[N_ID]) @ inv
                frac -= np.round(frac)
                times.append(ts * PS_PER_STEP)
                dists.append(float(np.linalg.norm(frac @ cell)))
    t = np.array(times)
    d = np.array(dists)
    np.savetxt(os.path.join(here, 'ganh_distance.csv'),
               np.column_stack([t, d]), delimiter=',',
               header='time_ps,d_GaadmolNadmol_A', comments='')
    print(len(t), 'samples,', t[0], '-', t[-1], 'ps')

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (t0, t1) in enumerate(DISSOC_INTERVALS):
        ax.axvspan(t0, t1, color='tab:orange', alpha=0.25, lw=0,
                   label='dissociated' if i == 0 else None)
    ax.plot(t, d, lw=0.5, color='tab:blue')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, d.max() * 1.28)  # headroom so the legend does not overlap the curve
    ax.set_xlabel('time (ps)')
    ax.set_ylabel(r'$d$(Ga$_\mathrm{admol}$–N$_\mathrm{admol}$) (Å)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    out = os.path.join(here, 'ganh_distance.pdf')
    plt.savefig(out)
    plt.savefig(out.replace('.pdf', '.png'), dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
