#!/usr/bin/env python3
"""
Extract, from a set of UMA-MD dump files, the minimum distance between Ga_admol
(id 184) and the topmost-layer Ga atoms (id 168-183) together with the
Ga_admol-N_admol (id 103) distance, and save them to CSV.

- Output: ga_topga_dmin.csv (time_ps, d_topGa_min, d_GaN)
- Sampling: every 100 steps (0.1 fs/step -> 10 fs)
- Distances use the minimum image convention for a triclinic cell
- Purpose: population analysis of the bonding states (lifted / surface-engaged)
  (source data for the 56%/44% split in the main text, using a 3.3 A Ga-Ga
  distance threshold)

Usage: place this script in the same directory as the *.dump files and run it.
"""
import glob
import itertools
import os
import re

import numpy as np

TOP_LAYER_GA_IDS = set(range(168, 184))   # topmost Ga layer (16 atoms)
GA_ADMOL_ID, N_ADMOL_ID = 184, 103
SAMPLE_EVERY = 100                        # steps
PS_PER_STEP = 1e-4


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
    ids = TOP_LAYER_GA_IDS | {GA_ADMOL_ID, N_ADMOL_ID}
    cell = inv = None
    rows = []
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
                b = [f.readline() for _ in range(3)]
                cols = f.readline().split()[2:]
                if ts % SAMPLE_EVERY or (rows and ts * PS_PER_STEP <= rows[-1][0]):
                    for _ in itertools.islice(f, natoms):
                        pass
                    continue
                if cell is None:
                    cell = box_to_cell(*b)
                    inv = np.linalg.inv(cell)
                ic = {c: i for i, c in enumerate(cols)}
                pos = {}
                for _ in range(natoms):
                    q = f.readline().split()
                    aid = int(q[ic['id']])
                    if aid in ids:
                        pos[aid] = np.array([float(q[ic['x']]),
                                             float(q[ic['y']]),
                                             float(q[ic['z']])])

                def mind(a, b_):
                    fr = (pos[a] - pos[b_]) @ inv
                    fr -= np.round(fr)
                    return np.linalg.norm(fr @ cell)

                dmin = min(mind(GA_ADMOL_ID, j) for j in TOP_LAYER_GA_IDS)
                rows.append((ts * PS_PER_STEP, dmin, mind(GA_ADMOL_ID, N_ADMOL_ID)))
    arr = np.array(rows)
    out = os.path.join(here, 'ga_topga_dmin.csv')
    np.savetxt(out, arr, delimiter=',',
               header='time_ps,d_topGa_min,d_GaN', comments='')
    print('wrote', out, f'({len(arr)} samples)')


if __name__ == '__main__':
    main()
