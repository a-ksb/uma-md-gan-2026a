from __future__ import annotations

"""t=0 の snapshot を UMA で構造最適化し、その最適化後座標から LAMMPS NVT-MD を回すワークフロー。

流れ:
1. data 由来の初期構造 (`ideal_0.vasp`) を読み込む。無い場合は明示エラーで終了する
2. UMA (`uma-m-1p1.pt`) で構造最適化する
3. 最適化後の座標を t=0 とみなして LAMMPS fix external + NVT を開始する
4. data 由来の初期速度 (`initial_velocities.txt`) を既定で用い、必要なら LAMMPS の乱数初期化に切り替える
5. 既にある最適化後構造をそのまま MD 開始点にする direct モードも `--optimized-structure` で指定できる
6. 生成物と要約を `results_long_reference_prep_20260327` 配下に保存する
"""

import argparse
import csv
import json
import math
import time
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Redirects are currently not supported in Windows or MacOs.",
)
for noisy_logger in (
    "torch",
    "torch.distributed",
    "torch.distributed.elastic",
    "torch.distributed.elastic.multiprocessing",
    "torch.distributed.elastic.multiprocessing.redirects",
):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import FIRE
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.lammps.lammps_fc import (
    FIX_EXT_ID,
    FIX_EXTERNAL_CMD,
    FixExternalCallback,
    check_input_script,
    separate_run_commands,
)
from lammps import lammps
from tqdm import tqdm

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Redirects are currently not supported in Windows or MacOs.",
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_INITIAL_STRUCTURE_PATH = REPO_ROOT / "ideal_0.vasp"
DEFAULT_INITIAL_VELOCITIES_PATH = REPO_ROOT / "initial_velocities.txt"
DEFAULT_MODEL_PATH = REPO_ROOT / "uma-m-1p1.pt"
DEFAULT_RUN_NAME = "fresh_oc20_relax_then_lammps_nvt_20260414"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results_long_reference_prep_20260327"
DEFAULT_TASK_NAME = "oc20"
DEFAULT_DEVICE = "cuda"
DEFAULT_TEMPERATURE_K = 1273.0
DEFAULT_TIMESTEP_FS = 0.1
DEFAULT_TDAMP_FS = 10.0
DEFAULT_STEPS = 15000
DEFAULT_FMAX = 0.05
DEFAULT_RELAX_MAX_STEPS = 500
DEFAULT_SKIP_RELAX = False
DEFAULT_VELOCITY_SEED = 12345
LAMMPS_DATA_FILENAME = "oc20_fix_external.data"
LAMMPS_INPUT_FILENAME = "oc20_fix_external.in"
LAMMPS_DUMP_FILENAME = "oc20_fix_external.dump"
LAMMPS_LOG_FILENAME = "lammps.log"
LAMMPS_PROGRESS_FILENAME = "md_progress.csv"
LAMMPS_RESTART_PATTERN = "oc20_fix_external.restart.*"
DEFAULT_FIXED_LAYER_COUNT = 0
DEFAULT_NVT_LAYER_START = 1
DEFAULT_NVT_LAYER_END = 4
DEFAULT_LAYER_GAP_THRESHOLD_A = 1.0
DEFAULT_FIXED_ATOM_IDS_1BASED = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
]


@dataclass(frozen=True)
class LayerGroupAssignment:
    rows: list[dict[str, object]]
    fixed_ids_1based: list[int]
    specified_fixed_ids_1based: list[int]
    auto_fixed_layer_ids_1based: list[int]
    bath_ids_1based: list[int]
    nve_region_ids_1based: list[int]
    movable_ids_1based: list[int]
    fixed_layer_count: int
    nvt_layer_start: int
    nvt_layer_end: int
    layer_gap_threshold: float
    detected_layer_count: int
    max_vacuum_gap_A: float
    slab_bottom_z_wrapped_A: float

    @property
    def group_counts(self) -> dict[str, object]:
        return {
            "fixed_atom_count": len(self.fixed_ids_1based),
            "specified_fixed_atom_count": len(self.specified_fixed_ids_1based),
            "auto_fixed_layer_atom_count": len(self.auto_fixed_layer_ids_1based),
            "bath_atom_count": len(self.bath_ids_1based),
            "nve_region_atom_count": len(self.nve_region_ids_1based),
            "movable_atom_count": len(self.movable_ids_1based),
            "fixed_layer_count": self.fixed_layer_count,
            "nvt_layer_start": self.nvt_layer_start,
            "nvt_layer_end": self.nvt_layer_end,
            "layer_gap_threshold": self.layer_gap_threshold,
            "detected_layer_count": self.detected_layer_count,
            "max_vacuum_gap_A": self.max_vacuum_gap_A,
            "slab_bottom_z_wrapped_A": self.slab_bottom_z_wrapped_A,
        }


@dataclass(frozen=True)
class WorkflowConfig:
    input_structure: Path
    optimized_structure: Path | None
    resume_from_run_dir: Path | None
    model_path: Path
    output_root: Path
    run_name: str
    output_layout: str
    task_name: str
    device: str
    temperature_k: float
    timestep_fs: float
    tdamp_fs: float
    thermostat_chain_length: int
    steps: int
    fmax: float
    relax_max_steps: int
    skip_relax: bool
    initial_velocities: Path | None
    random_initial_velocities: bool
    velocity_seed: int
    md_progress_interval: int
    restart_interval_steps: int | None
    print_summary: bool
    fixed_atom_ids_1based: list[int]
    fixed_layer_count: int
    nvt_layer_start: int
    nvt_layer_end: int
    layer_gap_threshold: float
    write_layer_assignment: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "WorkflowConfig":
        return cls(
            input_structure=args.input_structure,
            optimized_structure=args.optimized_structure,
            resume_from_run_dir=args.resume_from_run_dir,
            model_path=args.model_path,
            output_root=args.output_root,
            run_name=args.run_name,
            output_layout=args.output_layout,
            task_name=args.task_name,
            device=args.device,
            temperature_k=args.temperature_k,
            timestep_fs=args.timestep_fs,
            tdamp_fs=args.tdamp_fs,
            thermostat_chain_length=args.thermostat_chain_length,
            steps=args.steps,
            fmax=args.fmax,
            relax_max_steps=args.relax_max_steps,
            skip_relax=bool(args.skip_relax),
            initial_velocities=args.initial_velocities,
            random_initial_velocities=bool(args.random_initial_velocities),
            velocity_seed=args.velocity_seed,
            md_progress_interval=args.md_progress_interval,
            restart_interval_steps=None if args.restart_interval_steps is None else max(1, int(args.restart_interval_steps)),
            print_summary=bool(args.print_summary),
            fixed_atom_ids_1based=_resolve_fixed_atom_ids(args),
            fixed_layer_count=args.fixed_layer_count,
            nvt_layer_start=args.nvt_layer_start,
            nvt_layer_end=args.nvt_layer_end,
            layer_gap_threshold=args.layer_gap_threshold,
            write_layer_assignment=bool(args.write_layer_assignment),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize the t=0 snapshot with UMA, then run LAMMPS NVT-MD from the optimized structure."
    )
    parser.add_argument(
        "--input-structure",
        type=Path,
        default=DEFAULT_DATA_INITIAL_STRUCTURE_PATH,
        help=(
            "Input structure to optimize. Default is the data-derived ideal_0.vasp. "
            "If the file is missing, the workflow stops with FileNotFoundError."
        ),
    )
    parser.add_argument(
        "--optimized-structure",
        type=Path,
        default=None,
        help=(
            "Existing optimized structure to use as the MD start point. "
            "When set, the workflow skips UMA relaxation and copies this file to 01_optimized_t0.vasp."
        ),
    )
    parser.add_argument(
        "--resume-from-run-dir",
        type=Path,
        default=None,
        help=(
            "Existing flat run directory to resume from. "
            "If it contains oc20_fix_external.restart.* files, the workflow restarts from the latest binary checkpoint. "
            "If no restart file exists but oc20_fix_external.dump exists, it falls back to the latest dump snapshot."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--output-layout",
        choices=("nested", "flat"),
        default="nested",
        help=(
            "Output layout under output-root. "
            "'nested' writes to 03_runs/<run_name> and 04_outputs/<run_name>; "
            "'flat' writes all files directly under output-root. Use this for 1-300000, 300001-600000 chunk folders."
        ),
    )
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--timestep-fs", type=float, default=DEFAULT_TIMESTEP_FS)
    parser.add_argument("--tdamp-fs", type=float, default=DEFAULT_TDAMP_FS)
    parser.add_argument(
        "--thermostat-chain-length",
        type=int,
        default=5,
        help="Nose-Hoover chain length for LAMMPS NVT. CASTEP uses 5 in the reference MD file.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument("--relax-max-steps", type=int, default=DEFAULT_RELAX_MAX_STEPS)
    parser.add_argument("--skip-relax", action="store_true", help="Skip UMA relaxation and use the input structure as-is.")
    parser.add_argument(
        "--initial-velocities",
        type=Path,
        default=None,
        help=(
            "Optional per-atom velocity table. "
            "Each row may contain either 'vx vy vz' in atom order or 'atom_id vx vy vz'. "
            "When omitted, the workflow uses the data-derived initial_velocities.txt for the relax-from-data path. "
            "For --optimized-structure runs, this file is ignored and fresh Maxwell-Boltzmann velocities are generated in LAMMPS. "
            "Pass --random-initial-velocities to always ignore this file and generate Maxwell-Boltzmann velocities in LAMMPS."
        ),
    )
    parser.add_argument(
        "--random-initial-velocities",
        action="store_true",
        help="Ignore the data-derived initial_velocities.txt and generate Maxwell-Boltzmann velocities in LAMMPS.",
    )
    parser.add_argument(
        "--fixed-atom-ids-1based",
        type=int,
        nargs="+",
        default=None,
        help=(
            "1-based atom IDs to freeze. "
            "If omitted, the workflow uses the repository's default fixed-atom set."
        ),
    )
    parser.add_argument(
        "--fixed-atom-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file containing 1-based atom IDs to freeze, one row or whitespace-separated. "
            "Ignored when --fixed-atom-ids-1based is provided."
        ),
    )
    parser.add_argument(
        "--velocity-seed",
        type=int,
        default=DEFAULT_VELOCITY_SEED,
        help="Seed used when LAMMPS generates Maxwell-Boltzmann velocities for the mobile group.",
    )
    parser.add_argument(
        "--md-progress-interval",
        type=int,
        default=100,
        help="Write MD progress every N steps.",
    )
    parser.add_argument(
        "--restart-interval-steps",
        type=int,
        default=None,
        help=(
            "Write LAMMPS restart checkpoints every N steps. "
            "Defaults to the MD progress interval; the final step is always checkpointed when restart output is enabled."
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the final JSON summary to the terminal.",
    )
    parser.add_argument(
        "--fixed-layer-count",
        type=int,
        default=DEFAULT_FIXED_LAYER_COUNT,
        help=(
            "Number of physical bottom layers to add to the fixed group. "
            "Default 0 preserves only the explicit fixed atom IDs; pass 1 to fix the whole first layer."
        ),
    )
    parser.add_argument(
        "--nvt-layer-start",
        type=int,
        default=DEFAULT_NVT_LAYER_START,
        help="Bottom-based layer index where the NVT bath starts. Default: 1 for non-fixed substrate layers 1-4.",
    )
    parser.add_argument(
        "--nvt-layer-end",
        type=int,
        default=DEFAULT_NVT_LAYER_END,
        help="Bottom-based layer index where the NVT bath ends inclusively. Default: 4.",
    )
    parser.add_argument(
        "--layer-gap-threshold",
        type=float,
        default=DEFAULT_LAYER_GAP_THRESHOLD_A,
        help="Adjacent z_unwrapped gap in Å used as a layer boundary. Default: 1.0 Å.",
    )
    parser.add_argument(
        "--write-layer-assignment",
        dest="write_layer_assignment",
        action="store_true",
        default=True,
        help="Write layer_assignment.csv in the run directory. Enabled by default.",
    )
    parser.add_argument(
        "--no-write-layer-assignment",
        dest="write_layer_assignment",
        action="store_false",
        help="Do not write layer_assignment.csv. group_counts.json is still written.",
    )
    return parser


def _atomic_ids_to_zero_based(ids_1based: list[int]) -> list[int]:
    return [atom_id - 1 for atom_id in ids_1based]


def _format_fixed_group(ids_1based: list[int]) -> str:
    return " ".join(str(atom_id) for atom_id in ids_1based)


def _format_id_group(ids_1based: list[int]) -> str:
    if not ids_1based:
        raise ValueError("LAMMPS id group cannot be empty.")
    return " ".join(str(atom_id) for atom_id in ids_1based)


def _ensure_cuda_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def _resolve_fixed_atom_ids(args: argparse.Namespace) -> list[int]:
    if args.fixed_atom_ids_1based is not None:
        fixed_ids = [int(atom_id) for atom_id in args.fixed_atom_ids_1based]
    elif args.fixed_atom_ids_file is not None:
        text = args.fixed_atom_ids_file.read_text(encoding="utf-8-sig", errors="strict")
        tokens: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.split("#", 1)[0].strip()
            if not stripped:
                continue
            tokens.extend(stripped.replace(",", " ").split())
        fixed_ids = [int(token) for token in tokens]
    else:
        fixed_ids = list(DEFAULT_FIXED_ATOM_IDS_1BASED)

    if not fixed_ids:
        raise ValueError("At least one fixed atom ID is required.")
    if any(atom_id <= 0 for atom_id in fixed_ids):
        raise ValueError(f"Fixed atom IDs must be positive 1-based integers: {fixed_ids}")
    if len(set(fixed_ids)) != len(fixed_ids):
        raise ValueError(f"Fixed atom IDs contain duplicates: {fixed_ids}")
    return fixed_ids


def _validate_layer_parameters(
    fixed_layer_count: int,
    nvt_layer_start: int,
    nvt_layer_end: int,
    layer_gap_threshold: float,
) -> None:
    if fixed_layer_count < 0:
        raise ValueError(f"--fixed-layer-count must be >= 0, got {fixed_layer_count}.")
    if nvt_layer_start < 1:
        raise ValueError(f"--nvt-layer-start must be >= 1, got {nvt_layer_start}.")
    if nvt_layer_end < nvt_layer_start:
        raise ValueError(
            f"--nvt-layer-end must be >= --nvt-layer-start, got start={nvt_layer_start}, end={nvt_layer_end}."
        )
    if layer_gap_threshold <= 0:
        raise ValueError(f"--layer-gap-threshold must be positive, got {layer_gap_threshold}.")


def _assign_layers_and_groups(
    atoms: Atoms,
    fixed_ids_1based: list[int],
    fixed_layer_count: int,
    nvt_layer_start: int,
    nvt_layer_end: int,
    layer_gap_threshold: float,
) -> LayerGroupAssignment:
    """Assign fixed / NVT bath / NVE region from periodic-z-aware layer clustering."""

    _validate_layer_parameters(
        fixed_layer_count=fixed_layer_count,
        nvt_layer_start=nvt_layer_start,
        nvt_layer_end=nvt_layer_end,
        layer_gap_threshold=layer_gap_threshold,
    )
    natoms = len(atoms)
    if natoms <= 0:
        raise ValueError("Cannot assign layers for an empty Atoms object.")

    expected_ids = set(range(1, natoms + 1))
    specified_fixed_set = set(fixed_ids_1based)
    missing_fixed_ids = sorted(specified_fixed_set.difference(expected_ids))
    if missing_fixed_ids:
        raise ValueError(f"Fixed atom IDs exceed natoms={natoms}: {missing_fixed_ids[:10]}")

    cell = np.asarray(atoms.cell, dtype=float)
    c_length = float(np.linalg.norm(cell[2]))
    if not math.isfinite(c_length) or c_length <= 0:
        raise ValueError(f"Invalid cell c length for z wrapping: {c_length}")

    z_cart = np.asarray(atoms.positions, dtype=float)[:, 2]
    z_wrapped = np.mod(z_cart, c_length)
    sorted_by_wrapped = np.argsort(z_wrapped, kind="mergesort")
    z_sorted = z_wrapped[sorted_by_wrapped]
    gaps = np.diff(z_sorted)
    wrap_gap = (z_sorted[0] + c_length) - z_sorted[-1]
    cyclic_gaps = np.concatenate([gaps, np.array([wrap_gap], dtype=float)])
    max_gap_index = int(np.argmax(cyclic_gaps))
    max_vacuum_gap = float(cyclic_gaps[max_gap_index])
    min_vacuum_gap = max(2.0, 2.0 * float(layer_gap_threshold))
    if max_vacuum_gap < min_vacuum_gap:
        raise ValueError(
            "Maximum z gap is too small to identify the vacuum gap safely: "
            f"max_gap={max_vacuum_gap:.6f} Å, required>={min_vacuum_gap:.6f} Å, "
            f"layer_gap_threshold={layer_gap_threshold:.6f} Å."
        )

    slab_start_sorted_pos = (max_gap_index + 1) % natoms
    slab_start_atom_index = int(sorted_by_wrapped[slab_start_sorted_pos])
    slab_bottom_z_wrapped = float(z_wrapped[slab_start_atom_index])
    z_unwrapped = np.mod(z_wrapped - slab_bottom_z_wrapped, c_length)

    sorted_by_unwrapped = np.argsort(z_unwrapped, kind="mergesort")
    layer_by_atom_index = np.zeros(natoms, dtype=int)
    current_layer = 1
    previous_z = float(z_unwrapped[sorted_by_unwrapped[0]])
    layer_by_atom_index[sorted_by_unwrapped[0]] = current_layer
    for atom_index in sorted_by_unwrapped[1:]:
        this_z = float(z_unwrapped[atom_index])
        if this_z - previous_z >= layer_gap_threshold:
            current_layer += 1
        layer_by_atom_index[atom_index] = current_layer
        previous_z = this_z
    detected_layer_count = int(current_layer)
    if detected_layer_count < 5:
        raise ValueError(
            f"Detected only {detected_layer_count} layers; at least 5 are required for fixed/NVT/NVE splitting."
        )

    auto_fixed_layer_set = {
        atom_id
        for atom_id in expected_ids
        if int(layer_by_atom_index[atom_id - 1]) <= fixed_layer_count
    }
    fixed_set = specified_fixed_set.union(auto_fixed_layer_set)

    bath_set = {
        atom_id
        for atom_id in expected_ids
        if nvt_layer_start <= int(layer_by_atom_index[atom_id - 1]) <= nvt_layer_end
    }
    bath_set.difference_update(fixed_set)
    nve_region_set = expected_ids.difference(fixed_set).difference(bath_set)
    movable_set = bath_set.union(nve_region_set)

    if not bath_set:
        raise ValueError("bath group is empty. Check layer detection and --nvt-layer-start/end.")
    if not nve_region_set:
        raise ValueError("nve_region group is empty. Check layer detection and --nvt-layer-start/end.")
    if fixed_set & bath_set or fixed_set & nve_region_set or bath_set & nve_region_set:
        raise ValueError(
            "fixed, bath, and nve_region groups must be disjoint: "
            f"fixed∩bath={sorted(fixed_set & bath_set)[:10]}, "
            f"fixed∩nve={sorted(fixed_set & nve_region_set)[:10]}, "
            f"bath∩nve={sorted(bath_set & nve_region_set)[:10]}"
        )
    if fixed_set & bath_set or fixed_set & nve_region_set:
        raise ValueError("fixed atoms were mixed into bath or nve_region.")
    group_union = fixed_set.union(bath_set).union(nve_region_set)
    if group_union != expected_ids:
        missing = sorted(expected_ids.difference(group_union))
        extra = sorted(group_union.difference(expected_ids))
        raise ValueError(
            f"fixed + bath + nve_region does not cover all atoms: missing={missing[:10]}, extra={extra[:10]}"
        )

    symbols = atoms.get_chemical_symbols()
    rows: list[dict[str, object]] = []
    for atom_id in range(1, natoms + 1):
        if atom_id in fixed_set:
            group_assignment = "fixed"
            if atom_id in specified_fixed_set:
                assignment_source = "specified_fixed"
            else:
                assignment_source = "auto_fixed_layer"
        elif atom_id in bath_set:
            group_assignment = "bath"
            assignment_source = "bath"
        elif atom_id in nve_region_set:
            group_assignment = "nve_region"
            assignment_source = "nve_region"
        else:
            raise AssertionError(f"Atom {atom_id} was not assigned to any group.")
        atom_index = atom_id - 1
        rows.append(
            {
                "atom_id": atom_id,
                "element": symbols[atom_index],
                "z_wrapped": float(z_wrapped[atom_index]),
                "z_unwrapped": float(z_unwrapped[atom_index]),
                "layer_index_from_bottom": int(layer_by_atom_index[atom_index]),
                "group_assignment": group_assignment,
                "assignment_source": assignment_source,
            }
        )

    return LayerGroupAssignment(
        rows=rows,
        fixed_ids_1based=sorted(fixed_set),
        specified_fixed_ids_1based=sorted(specified_fixed_set),
        auto_fixed_layer_ids_1based=sorted(auto_fixed_layer_set.difference(specified_fixed_set)),
        bath_ids_1based=sorted(bath_set),
        nve_region_ids_1based=sorted(nve_region_set),
        movable_ids_1based=sorted(movable_set),
        fixed_layer_count=fixed_layer_count,
        nvt_layer_start=nvt_layer_start,
        nvt_layer_end=nvt_layer_end,
        layer_gap_threshold=float(layer_gap_threshold),
        detected_layer_count=detected_layer_count,
        max_vacuum_gap_A=max_vacuum_gap,
        slab_bottom_z_wrapped_A=slab_bottom_z_wrapped,
    )


def _write_layer_assignment_csv(path: Path, assignment: LayerGroupAssignment) -> None:
    fieldnames = [
        "atom_id",
        "element",
        "z_wrapped",
        "z_unwrapped",
        "layer_index_from_bottom",
        "group_assignment",
        "assignment_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignment.rows)


def _write_group_counts_json(path: Path, assignment: LayerGroupAssignment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(assignment.group_counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_atoms(path: Path):
    atoms = read(path)
    atoms.pbc = True
    return atoms


def _load_reference_initial_structure() -> tuple[Atoms, dict[str, object]]:
    """Load the data-derived initial structure.

    The repository is expected to keep `ideal_0.vasp` alongside the workflow.
    If it is missing, fail fast instead of silently changing the initial state.
    """

    if DEFAULT_DATA_INITIAL_STRUCTURE_PATH.exists():
        return _load_atoms(DEFAULT_DATA_INITIAL_STRUCTURE_PATH), {
            "mode": "existing_file",
            "path": str(DEFAULT_DATA_INITIAL_STRUCTURE_PATH),
            "required": True,
        }

    raise FileNotFoundError(
        f"Default initial structure {DEFAULT_DATA_INITIAL_STRUCTURE_PATH} is required but was not found. "
        "Restore it before running the workflow so the MD starts from the intended data-derived structure."
    )


def _snapshot_signature(atoms) -> dict[str, object]:
    return {
        "natoms": len(atoms),
        "cell_det_A3": float(abs(np.linalg.det(np.asarray(atoms.cell, dtype=float)))),
        "position_checksum": float(np.asarray(atoms.positions, dtype=float).sum()),
    }


def _load_lammps_dump_snapshot(path: Path) -> Atoms:
    atoms = read(path, index=-1, format="lammps-dump-text")
    atoms.pbc = True
    return atoms


def _write_velocity_table_from_atoms(atoms: Atoms, path: Path) -> None:
    velocities = np.asarray(atoms.get_velocities(), dtype=float)
    if velocities.size == 0:
        raise ValueError("The provided atoms object does not contain velocities.")
    lines = [f"{vx:.16e} {vy:.16e} {vz:.16e}" for vx, vy, vz in velocities]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json_text(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _restart_sort_key(path: Path) -> tuple[int, float, str]:
    marker = ".restart."
    name = path.name
    if marker in name:
        suffix = name.rsplit(marker, 1)[-1]
        if suffix.isdigit():
            return (1, float(int(suffix)), name)
    return (0, path.stat().st_mtime, name)


def _validate_exact_resume_compatibility(previous_summary: dict[str, object], config: WorkflowConfig) -> None:
    run_params = previous_summary.get("run_parameters", {})
    relax_meta = previous_summary.get("relax", {})
    mismatches: list[str] = []

    expected_fixed = relax_meta.get("fixed_atom_ids_1based")
    if expected_fixed is not None and list(expected_fixed) != list(config.fixed_atom_ids_1based):
        mismatches.append("fixed_atom_ids_1based")

    scalar_checks = [
        ("temperature_k", config.temperature_k),
        ("timestep_fs", config.timestep_fs),
        ("tdamp_fs", config.tdamp_fs),
        ("thermostat_chain_length", config.thermostat_chain_length),
        ("fixed_layer_count", config.fixed_layer_count),
        ("nvt_layer_start", config.nvt_layer_start),
        ("nvt_layer_end", config.nvt_layer_end),
    ]
    for key, expected in scalar_checks:
        actual = run_params.get(key)
        if actual is not None and actual != expected:
            mismatches.append(key)

    if mismatches:
        raise ValueError(
            "Exact restart resume was requested, but the resume source parameters differ from the current run: "
            + ", ".join(mismatches)
            + ". Use the same fixed atoms / layer split / timestep / thermostat settings as the source run."
        )


def _parse_velocity_table(path: Path, natoms: int) -> dict[int, tuple[float, float, float]]:
    """Read a simple velocity table.

    Accepted formats:
    - `vx vy vz`
    - `atom_id vx vy vz`

    Blank lines and `#` comments are ignored. When atom IDs are omitted, rows are
    mapped in atom order from 1..natoms.
    """

    velocities_by_id: dict[int, tuple[float, float, float]] = {}
    positional_rows: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="strict").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) == 3:
            vx, vy, vz = (float(value) for value in parts)
            positional_rows.append((vx, vy, vz))
            continue
        if len(parts) == 4:
            atom_id = int(parts[0])
            vx, vy, vz = (float(value) for value in parts[1:])
            velocities_by_id[atom_id] = (vx, vy, vz)
            continue
        raise ValueError(
            f"Unsupported velocity row in {path}: expected 3 or 4 columns, got {len(parts)}"
        )

    if velocities_by_id and positional_rows:
        raise ValueError(f"Mixed velocity formats are not allowed in {path}. Use either atom-id rows or positional rows.")

    if velocities_by_id:
        expected_ids = set(range(1, natoms + 1))
        missing = expected_ids.difference(velocities_by_id)
        extra = set(velocities_by_id).difference(expected_ids)
        if missing or extra:
            raise ValueError(
                f"Velocity table {path} does not match natoms={natoms}: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )
        return velocities_by_id

    if len(positional_rows) != natoms:
        raise ValueError(f"Velocity table {path} has {len(positional_rows)} rows but natoms={natoms}.")

    return {atom_id: positional_rows[atom_id - 1] for atom_id in range(1, natoms + 1)}


def _build_velocity_initialization_lines(
    natoms: int,
    fixed_ids_1based: list[int],
    temperature_k: float,
    velocity_seed: int,
    initial_velocities_path: Path | None,
) -> tuple[list[str], dict[str, object]]:
    """Build the initial-velocity section of the LAMMPS input."""

    if initial_velocities_path is not None:
        velocities_by_id = _parse_velocity_table(initial_velocities_path, natoms)
        velocity_lines: list[str] = []
        for atom_id in range(1, natoms + 1):
            vx, vy, vz = velocities_by_id[atom_id]
            velocity_lines.append(f"set atom {atom_id} vx {vx:.16e} vy {vy:.16e} vz {vz:.16e}")
        for atom_id in fixed_ids_1based:
            velocity_lines.append(f"set atom {atom_id} vx 0.0 vy 0.0 vz 0.0")
        return velocity_lines, {
            "mode": "external_file",
            "seed": None,
            "source_path": str(initial_velocities_path),
        }

    return [
        f"velocity movable create {temperature_k:.1f} {velocity_seed} dist gaussian mom yes rot yes",
        "velocity movable zero linear",
    ], {
        "mode": "movable_create",
        "seed": velocity_seed,
        "source_path": None,
    }


def optimize_with_uma(
    atoms,
    model_path: Path,
    task_name: str,
    device: str,
    fixed_ids_1based: list[int],
    fmax: float,
    relax_max_steps: int,
    output_dir: Path,
):
    device = _ensure_cuda_device(device)
    calculator = FAIRChemCalculator.from_model_checkpoint(str(model_path), task_name=task_name, device=device)
    relax_atoms = atoms.copy()
    relax_atoms.calc = calculator
    relax_atoms.set_constraint(FixAtoms(indices=_atomic_ids_to_zero_based(fixed_ids_1based)))

    initial_energy = float(relax_atoms.get_potential_energy())
    initial_forces = np.asarray(relax_atoms.get_forces(), dtype=float)
    initial_max_force = float(np.max(np.linalg.norm(initial_forces, axis=1)))

    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = FIRE(relax_atoms, logfile=str(output_dir / "relax_fire.log"))
    optimizer.run(fmax=fmax, steps=relax_max_steps)

    relaxed_energy = float(relax_atoms.get_potential_energy())
    relaxed_forces = np.asarray(relax_atoms.get_forces(), dtype=float)
    relaxed_max_force = float(np.max(np.linalg.norm(relaxed_forces, axis=1)))
    return relax_atoms, {
        "device": device,
        "model_path": str(model_path),
        "task_name": task_name,
        "initial_energy_eV": initial_energy,
        "initial_max_force_eV_per_A": initial_max_force,
        "relaxed_energy_eV": relaxed_energy,
        "relaxed_max_force_eV_per_A": relaxed_max_force,
        "relax_fmax_eV_per_A": fmax,
        "relax_max_steps": relax_max_steps,
        "fixed_atom_ids_1based": fixed_ids_1based,
        "snapshot_signature": _snapshot_signature(relax_atoms),
    }


def write_lammps_input(
    path: Path,
    data_path: Path,
    dump_path: Path,
    group_assignment: LayerGroupAssignment,
    temperature_k: float,
    velocity_seed: int,
    timestep_fs: float,
    tdamp_fs: float,
    thermostat_chain_length: int,
    thermo_interval: int,
    steps: int,
    natoms: int,
    initial_velocities_path: Path | None,
    restart_source_path: Path | None = None,
) -> None:
    timestep_ps = timestep_fs * 0.001
    tdamp_ps = tdamp_fs * 0.001
    data_file = data_path.resolve().as_posix()
    dump_file = dump_path.resolve().as_posix()
    if restart_source_path is None:
        velocity_lines, _velocity_meta = _build_velocity_initialization_lines(
            natoms=natoms,
            fixed_ids_1based=group_assignment.fixed_ids_1based,
            temperature_k=temperature_k,
            velocity_seed=velocity_seed,
            initial_velocities_path=initial_velocities_path,
        )
        prefix_lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f'read_data "{data_file}"',
        ]
    else:
        velocity_lines = []
        prefix_lines = [
            f'read_restart "{restart_source_path.as_posix()}"',
        ]
    text = "\n".join(
        [
            *prefix_lines,
            "neighbor 2.0 bin",
            "neigh_modify delay 0 every 1 check yes",
            "thermo_style custom step temp pe ke etotal press",
            f"thermo {thermo_interval}",
            f'dump oc20 all custom 1 "{dump_file}" id type x y z vx vy vz fx fy fz',
            "dump_modify oc20 sort id",
            f"timestep {timestep_ps:.10f}",
            f"group fixed id {_format_id_group(group_assignment.fixed_ids_1based)}",
            f"group bath id {_format_id_group(group_assignment.bath_ids_1based)}",
            "group nve_region subtract all fixed bath",
            "group movable union bath nve_region",
            *velocity_lines,
            "compute bath_temp bath temp",
            "thermo_modify temp bath_temp",
            "fix hold_fixed fixed setforce 0.0 0.0 0.0",
            "velocity fixed set 0.0 0.0 0.0",
            f"fix int_bath bath nvt temp {temperature_k:.1f} {temperature_k:.1f} {tdamp_ps:.10f} tchain {thermostat_chain_length}",
            "fix int_nve nve_region nve",
            f"run {steps}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _parse_run_steps(run_cmds: list[str]) -> int:
    total = 0
    for cmd in run_cmds:
        parts = cmd.split()
        if len(parts) >= 2 and parts[0] == "run":
            total += int(parts[1])
    return total


def run_lammps_with_fairchem_logged(
    predictor,
    lammps_input_path: str,
    task_name: str,
    log_path: Path,
    progress_log_path: Path,
    progress_interval: int,
    restart_output_pattern: Path | None = None,
    restart_interval_steps: int | None = None,
    charge: int = 0,
    spin: int = 0,
):
    machine = None
    if "LAMMPS_MACHINE_NAME" in os.environ:
        machine = os.environ["LAMMPS_MACHINE_NAME"]
    lmp = lammps(name=machine, cmdargs=["-nocite", "-log", str(log_path), "-screen", "none", "-echo", "none"])
    lmp._predictor = predictor
    lmp._task_name = task_name
    with open(lammps_input_path, encoding="utf-8") as f:
        input_script = f.read()
        check_input_script(input_script)
        script, run_cmds = separate_run_commands(input_script)
        lmp.commands_list(script)
        lmp.command(FIX_EXTERNAL_CMD)
        fix_external_call_back = FixExternalCallback(charge=charge, spin=spin)
        lmp.set_fix_external_callback(FIX_EXT_ID, fix_external_call_back, lmp)
        total_steps = _parse_run_steps(run_cmds)
        if total_steps <= 0:
            raise ValueError("LAMMPS input does not contain a valid run command.")

        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        progress_log_path.write_text(
            "completed_step,wall_time_s,temp,pe,ke,etotal,press\n",
            encoding="utf-8",
        )
        chunk_size = max(1, int(progress_interval))
        checkpoint_interval = chunk_size if restart_interval_steps is None else max(1, int(restart_interval_steps))
        checkpoint_accum = 0
        restart_path_text = restart_output_pattern.as_posix() if restart_output_pattern is not None else None
        started = time.perf_counter()
        completed = 0
        with tqdm(total=total_steps, desc="LAMMPS MD", unit="step", dynamic_ncols=True) as pbar:
            while completed < total_steps:
                this_chunk = min(chunk_size, total_steps - completed)
                lmp.command(f"run {this_chunk}")
                completed += this_chunk
                checkpoint_accum += this_chunk
                wall_time_s = time.perf_counter() - started
                current_step = int(round(float(lmp.get_thermo("step"))))
                row = [
                    str(current_step),
                    f"{wall_time_s:.3f}",
                ]
                for key in ("temp", "pe", "ke", "etotal", "press"):
                    try:
                        row.append(f"{float(lmp.get_thermo(key)):.10f}")
                    except Exception:
                        row.append("")
                with progress_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(",".join(row) + "\n")
                pbar.update(this_chunk)
                try:
                    temp_val = float(lmp.get_thermo("temp"))
                    pe_val = float(lmp.get_thermo("pe"))
                    pbar.set_postfix_str(f"step={current_step} (+{completed}/{total_steps}) T={temp_val:.1f}K PE={pe_val:.3f}eV")
                except Exception:
                    pbar.set_postfix_str(f"step={current_step} (+{completed}/{total_steps})")
                if restart_path_text is not None and (checkpoint_accum >= checkpoint_interval or completed == total_steps):
                    lmp.command(f'write_restart "{restart_path_text}"')
                    checkpoint_accum = 0
    return lmp


def run_workflow(config: WorkflowConfig) -> dict[str, object]:
    workflow_start = time.perf_counter()
    workflow_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if config.output_layout == "flat":
        run_dir = config.output_root
        out_dir = config.output_root
    else:
        run_dir = config.output_root / "03_runs" / config.run_name
        out_dir = config.output_root / "04_outputs" / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_source_run_dir = config.resume_from_run_dir
    resume_mode = "fresh"
    resume_source_summary_path: Path | None = None
    resume_source_dump_path: Path | None = None
    resume_source_restart_path: Path | None = None
    resume_source_atoms: Atoms | None = None
    resume_source_summary: dict[str, object] | None = None
    resume_exact_restart = False

    if resume_source_run_dir is not None:
        if not resume_source_run_dir.exists():
            raise FileNotFoundError(f"Resume source run dir {resume_source_run_dir} does not exist.")
        resume_source_dump_path = resume_source_run_dir / LAMMPS_DUMP_FILENAME
        if not resume_source_dump_path.exists():
            raise FileNotFoundError(
                f"Resume source dump {resume_source_dump_path} does not exist. "
                "The workflow needs the previous dump to rebuild a resume snapshot and verify atom ordering."
            )
        resume_source_atoms = _load_lammps_dump_snapshot(resume_source_dump_path)
        resume_source_summary_path = resume_source_run_dir / "summary.json"
        if resume_source_summary_path.exists():
            resume_source_summary = _load_json_text(resume_source_summary_path)
        restart_candidates = sorted(resume_source_run_dir.glob(LAMMPS_RESTART_PATTERN), key=_restart_sort_key)
        if restart_candidates:
            resume_mode = "exact_restart"
            resume_exact_restart = True
            resume_source_restart_path = restart_candidates[-1]
            if resume_source_summary is not None:
                _validate_exact_resume_compatibility(resume_source_summary, config)
        else:
            resume_mode = "dump_snapshot"

    md_start_source_path: Path
    md_start_structure_mode: str

    if resume_source_atoms is not None:
        atoms = resume_source_atoms
        initial_structure_meta = {
            "mode": resume_mode,
            "path": str(resume_source_dump_path),
            "required": True,
            "source_run_dir": str(resume_source_run_dir),
        }
        md_start_source_path = resume_source_dump_path if resume_source_dump_path is not None else resume_source_run_dir
        md_start_structure_mode = resume_mode
        relax_is_skipped = True
    elif config.optimized_structure is not None:
        relax_is_skipped = True
        if not config.optimized_structure.exists():
            raise FileNotFoundError(
                f"Optimized structure {config.optimized_structure} does not exist. "
                "The workflow will not auto-reconstruct a substitute."
            )
        atoms = _load_atoms(config.optimized_structure)
        initial_structure_meta = {
            "mode": "preoptimized_file",
            "path": str(config.optimized_structure),
            "required": True,
        }
        md_start_source_path = config.optimized_structure
        md_start_structure_mode = "preoptimized_file"
    elif config.input_structure == DEFAULT_DATA_INITIAL_STRUCTURE_PATH:
        relax_is_skipped = bool(config.skip_relax)
        atoms, initial_structure_meta = _load_reference_initial_structure()
        md_start_source_path = config.input_structure
        md_start_structure_mode = "relaxed_from_data_input"
    else:
        relax_is_skipped = bool(config.skip_relax)
        if not config.input_structure.exists():
            raise FileNotFoundError(
                f"Input structure {config.input_structure} does not exist. "
                "The workflow will not auto-reconstruct a substitute."
            )
        atoms = _load_atoms(config.input_structure)
        initial_structure_meta = {
            "mode": "user_supplied",
            "path": str(config.input_structure),
            "required": True,
        }
        md_start_source_path = config.input_structure
        md_start_structure_mode = "relaxed_from_user_input"
    initial_vasp_path = out_dir / "00_initial_structure.vasp"
    write(initial_vasp_path, atoms, format="vasp")

    relax_started = time.perf_counter()
    if relax_is_skipped:
        relaxed_atoms = atoms.copy()
        relax_meta = {
            "skipped": True,
            "device": _ensure_cuda_device(config.device),
            "model_path": str(config.model_path),
            "task_name": config.task_name,
            "initial_energy_eV": None,
            "initial_max_force_eV_per_A": None,
            "relaxed_energy_eV": None,
            "relaxed_max_force_eV_per_A": None,
            "relax_fmax_eV_per_A": config.fmax,
            "relax_max_steps": 0,
            "fixed_atom_ids_1based": config.fixed_atom_ids_1based,
            "snapshot_signature": _snapshot_signature(relaxed_atoms),
            "source_path": str(md_start_source_path),
            "mode": md_start_structure_mode,
        }
        if resume_source_run_dir is not None:
            relax_meta["resume_source_run_dir"] = str(resume_source_run_dir)
            relax_meta["resume_mode"] = resume_mode
    else:
        relaxed_atoms, relax_meta = optimize_with_uma(
            atoms=atoms,
            model_path=config.model_path,
            task_name=config.task_name,
            device=config.device,
            fixed_ids_1based=config.fixed_atom_ids_1based,
            fmax=config.fmax,
            relax_max_steps=config.relax_max_steps,
            output_dir=out_dir,
        )
    relax_wall_time_s = time.perf_counter() - relax_started

    relaxed_vasp_path = out_dir / "01_optimized_t0.vasp"
    write(relaxed_vasp_path, relaxed_atoms, format="vasp")

    data_path = run_dir / LAMMPS_DATA_FILENAME
    input_path = run_dir / LAMMPS_INPUT_FILENAME
    dump_path = run_dir / LAMMPS_DUMP_FILENAME
    lammps_log_path = run_dir / LAMMPS_LOG_FILENAME
    progress_log_path = run_dir / LAMMPS_PROGRESS_FILENAME
    restart_pattern_path = run_dir / LAMMPS_RESTART_PATTERN
    layer_assignment_path = run_dir / "layer_assignment.csv"
    group_counts_path = run_dir / "group_counts.json"

    layer_group_assignment = _assign_layers_and_groups(
        atoms=relaxed_atoms,
        fixed_ids_1based=config.fixed_atom_ids_1based,
        fixed_layer_count=config.fixed_layer_count,
        nvt_layer_start=config.nvt_layer_start,
        nvt_layer_end=config.nvt_layer_end,
        layer_gap_threshold=config.layer_gap_threshold,
    )
    if config.write_layer_assignment:
        _write_layer_assignment_csv(layer_assignment_path, layer_group_assignment)
    _write_group_counts_json(group_counts_path, layer_group_assignment)

    write(
        data_path,
        relaxed_atoms,
        format="lammps-data",
        specorder=["Ga", "N", "H"],
        masses=True,
        atom_style="atomic",
        units="metal",
        force_skew=True,
    )
    resume_velocity_table_path: Path | None = None
    if resume_source_run_dir is not None and not resume_exact_restart:
        resume_velocity_table_path = out_dir / "00_resume_velocities.txt"
        _write_velocity_table_from_atoms(relaxed_atoms, resume_velocity_table_path)

    if resume_source_run_dir is not None and resume_exact_restart:
        effective_initial_velocities_path = None
    elif resume_source_run_dir is not None:
        effective_initial_velocities_path = resume_velocity_table_path if resume_velocity_table_path is not None else None
    elif config.random_initial_velocities:
        effective_initial_velocities_path = None
    elif config.optimized_structure is not None:
        effective_initial_velocities_path = None
    else:
        effective_initial_velocities_path = config.initial_velocities or DEFAULT_INITIAL_VELOCITIES_PATH
        if not effective_initial_velocities_path.exists():
            raise FileNotFoundError(
                f"Initial velocities file {effective_initial_velocities_path} was not found. "
                "Restore the data-derived file or pass --random-initial-velocities explicitly."
            )

    write_lammps_input(
        path=input_path,
        data_path=data_path,
        dump_path=dump_path,
        group_assignment=layer_group_assignment,
        temperature_k=config.temperature_k,
        velocity_seed=config.velocity_seed,
        timestep_fs=config.timestep_fs,
        tdamp_fs=config.tdamp_fs,
        thermostat_chain_length=config.thermostat_chain_length,
        thermo_interval=config.md_progress_interval,
        steps=config.steps,
        natoms=len(relaxed_atoms),
        initial_velocities_path=effective_initial_velocities_path,
        restart_source_path=resume_source_restart_path if resume_exact_restart else None,
    )

    if resume_source_run_dir is not None and resume_exact_restart:
        velocity_mode = "restart_state"
    elif resume_source_run_dir is not None:
        velocity_mode = "dump_snapshot"
    elif effective_initial_velocities_path is not None:
        velocity_mode = "external_file"
    else:
        velocity_mode = "movable_create"

    predictor = load_predict_unit(
        path=str(config.model_path),
        inference_settings="default",
        device=relax_meta["device"],
    )
    lammps_started = time.perf_counter()
    lmp = run_lammps_with_fairchem_logged(
        predictor=predictor,
        lammps_input_path=str(input_path),
        task_name=config.task_name,
        log_path=lammps_log_path,
        progress_log_path=progress_log_path,
        progress_interval=config.md_progress_interval,
        restart_output_pattern=restart_pattern_path,
        restart_interval_steps=config.restart_interval_steps,
        charge=0,
        spin=0,
    )
    lammps_wall_time_s = time.perf_counter() - lammps_started

    thermo_keys = ["step", "temp", "pe", "ke", "etotal", "press"]
    final_thermo = {}
    for key in thermo_keys:
        try:
            final_thermo[key] = float(lmp.get_thermo(key))
        except Exception:
            final_thermo[key] = None

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report_ja.md"

    resume_summary = {
        "enabled": bool(resume_source_run_dir is not None),
        "mode": resume_mode if resume_source_run_dir is not None else None,
        "source_run_dir": str(resume_source_run_dir) if resume_source_run_dir is not None else None,
        "source_dump_path": str(resume_source_dump_path) if resume_source_dump_path is not None else None,
        "source_restart_path": str(resume_source_restart_path) if resume_source_restart_path is not None else None,
        "source_summary_path": str(resume_source_summary_path) if resume_source_summary_path is not None else None,
    }

    summary = {
        "workflow_started_at": workflow_started_at,
        "run_name": config.run_name,
        "input_structure": str(resume_source_dump_path)
        if resume_source_dump_path is not None
        else str(config.optimized_structure)
        if config.optimized_structure is not None
        else str(config.input_structure)
        if config.input_structure is not None
        else None,
        "optimized_structure": str(config.optimized_structure) if config.optimized_structure is not None else None,
        "resume": resume_summary,
        "input_structure_meta": initial_structure_meta,
        "initial_structure": str(initial_vasp_path),
        "relaxed_structure": str(relaxed_vasp_path),
        "md_start_structure": str(relaxed_vasp_path),
        "md_start_structure_meta": {
            "mode": resume_mode if resume_source_run_dir is not None else ("preoptimized_file" if config.optimized_structure is not None else "relaxed_structure"),
            "source_path": str(md_start_source_path),
            "path": str(relaxed_vasp_path),
        },
        "output_dir": str(out_dir),
        "run_dir": str(run_dir),
        "output_layout": config.output_layout,
        "files": {
            "data_path": str(data_path),
            "input_script_path": str(input_path),
            "dump_path": str(dump_path),
            "log_path": str(lammps_log_path),
            "progress_log_path": str(progress_log_path),
            "restart_pattern": str(restart_pattern_path),
            "layer_assignment_path": str(layer_assignment_path) if config.write_layer_assignment else None,
            "group_counts_path": str(group_counts_path),
        },
        "relax": relax_meta,
        "timing": {
            "relax_wall_time_s": relax_wall_time_s,
            "lammps_wall_time_s": lammps_wall_time_s,
            "total_wall_time_s": time.perf_counter() - workflow_start,
        },
        "run_parameters": {
            "temperature_k": config.temperature_k,
            "timestep_fs": config.timestep_fs,
            "tdamp_fs": config.tdamp_fs,
            "thermostat_chain_length": config.thermostat_chain_length,
            "thermostat_mode": "bath_nvt_plus_nve_region_nve",
            "steps": config.steps,
            "md_progress_interval": config.md_progress_interval,
            "restart_interval_steps": config.restart_interval_steps if config.restart_interval_steps is not None else config.md_progress_interval,
            "device": relax_meta["device"],
            "fixed_atom_count": len(layer_group_assignment.fixed_ids_1based),
            "bath_atom_count": len(layer_group_assignment.bath_ids_1based),
            "nve_region_atom_count": len(layer_group_assignment.nve_region_ids_1based),
            "movable_atom_count": len(layer_group_assignment.movable_ids_1based),
            "fixed_layer_count": config.fixed_layer_count,
            "nvt_layer_start": config.nvt_layer_start,
            "nvt_layer_end": config.nvt_layer_end,
            "layer_gap_threshold": config.layer_gap_threshold,
            "detected_layer_count": layer_group_assignment.detected_layer_count,
            "max_vacuum_gap_A": layer_group_assignment.max_vacuum_gap_A,
            "skip_relax": bool(relax_is_skipped),
            "resume_mode": resume_mode if resume_source_run_dir is not None else None,
            "initial_velocity_mode": velocity_mode,
            "velocity_seed": config.velocity_seed if velocity_mode == "movable_create" else None,
            "initial_velocities_path": str(effective_initial_velocities_path) if effective_initial_velocities_path is not None else None,
        },
        "layer_groups": layer_group_assignment.group_counts,
        "final_thermo": final_thermo,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_lines = [
        f"# {config.run_name} レポート",
        "",
        "## 目的",
        "- 前回 chunk を引き継いで、同じ 1-4 層 NVT / 上層 NVE 条件のまま LAMMPS MD を続きから実行する。" if resume_source_run_dir is not None else "- 最適化後座標から、1-4 層 NVT / 上層 NVE の LAMMPS fix external MD を開始する。",
        "",
        "## 入力",
        f"- 初期構造: `{initial_structure_meta['path']}`",
        f"- 初期構造モード: `{initial_structure_meta['mode']}`",
        "" if config.optimized_structure is None else f"- 既存最適化構造: `{config.optimized_structure}`",
        f"- UMA checkpoint: `{config.model_path}`",
        f"- UMA device: `{relax_meta['device']}`",
        "" if resume_source_run_dir is None else f"- 再開元 run dir: `{resume_source_run_dir}`",
        "" if resume_source_restart_path is None else f"- 再開元 restart file: `{resume_source_restart_path}`",
        "",
        "## UMA 最適化",
        "- 既存最適化構造を MD 開始点として再利用する。" if config.optimized_structure is not None else ("- relax をスキップして入力構造をそのまま初期条件に使う。" if config.skip_relax or resume_source_run_dir is not None else f"- 初期エネルギー: {relax_meta['initial_energy_eV']:.6f} eV"),
        "" if relax_is_skipped else f"- 最適化後エネルギー: {relax_meta['relaxed_energy_eV']:.6f} eV",
        "" if relax_is_skipped else f"- エネルギー差: {relax_meta['initial_energy_eV'] - relax_meta['relaxed_energy_eV']:.6f} eV",
        "" if relax_is_skipped else f"- 初期最大力: {relax_meta['initial_max_force_eV_per_A']:.6f} eV/Å",
        "" if relax_is_skipped else f"- 最適化後最大力: {relax_meta['relaxed_max_force_eV_per_A']:.6f} eV/Å",
        "",
        "## LAMMPS NVT-MD",
        f"- target temperature: {config.temperature_k:.1f} K",
        f"- timestep: {config.timestep_fs:.3f} fs",
        f"- tdamp: {config.tdamp_fs:.3f} fs",
        f"- thermostat chain length: {config.thermostat_chain_length}",
        f"- steps: {config.steps}",
        f"- fixed atom count: {len(layer_group_assignment.fixed_ids_1based)}",
        f"- bath atom count: {len(layer_group_assignment.bath_ids_1based)}",
        f"- nve_region atom count: {len(layer_group_assignment.nve_region_ids_1based)}",
        f"- detected layer count: {layer_group_assignment.detected_layer_count}",
        f"- NVT bath layers: {config.nvt_layer_start}–{config.nvt_layer_end}",
        f"- max vacuum gap: {layer_group_assignment.max_vacuum_gap_A:.6f} Å",
        f"- initial velocity mode: {velocity_mode}",
        f"- restart checkpoint interval: {summary['run_parameters']['restart_interval_steps']} step",
        f"- MD start structure: `{relaxed_vasp_path}`",
        "" if effective_initial_velocities_path is None else f"- initial velocities file: `{effective_initial_velocities_path}`",
        "" if effective_initial_velocities_path is not None else f"- velocity seed: {config.velocity_seed}",
        "",
        "## 出力",
        f"- output layout: `{config.output_layout}`",
        f"- optimized vasp: `{relaxed_vasp_path}`",
        f"- LAMMPS data: `{data_path}`",
        f"- LAMMPS input: `{input_path}`",
        f"- dump: `{dump_path}`",
        f"- restart pattern: `{restart_pattern_path}`",
        f"- progress log: `{progress_log_path}`",
        "" if not config.write_layer_assignment else f"- layer assignment: `{layer_assignment_path}`",
        f"- group counts: `{group_counts_path}`",
        f"- summary: `{summary_path}`",
        f"- relax wall time: {relax_wall_time_s:.2f} s",
        f"- LAMMPS wall time: {lammps_wall_time_s:.2f} s",
        f"- total wall time: {summary['timing']['total_wall_time_s']:.2f} s",
        "",
        "## 備考",
        (
            "- LAMMPS の初期速度は再開元の restart state をそのまま引き継いでいる。"
            if velocity_mode == "restart_state"
            else (
                "- LAMMPS の初期速度は dump snapshot の速度を再投入している。"
                if velocity_mode == "dump_snapshot"
                else (
                    "- LAMMPS の初期速度は外部速度テーブルを適用している。"
                    if velocity_mode == "external_file"
                    else "- LAMMPS の初期速度は target temperature から与えている。"
                )
            )
        ),
        "- `--output-layout flat` では `--output-root` 直下に summary / dump / restart / layer_assignment をすべて直接出力する。",
        "- `--fixed-layer-count 0` は、`fixed_atom_ids_1based.txt` または CLI 指定の固定IDのみを固定する。",
        "- `--fixed-layer-count 1` は、固定IDに加えて物理的な第1層全体を固定する。",
        f"- 今回の fixed_layer_count: {config.fixed_layer_count}",
        f"- 第{config.nvt_layer_start}〜{config.nvt_layer_end}層相当の bath のみ `fix nvt`、それ以外の可動領域 nve_region は `fix nve` に分けている。",
        "- 固定原子には `velocity fixed set 0 0 0` と `fix setforce 0 0 0` を入れ、NVT/NVE の時間積分 fix には含めていない。",
        "- 再開元に binary restart があれば exact restart、無ければ dump スナップショットからの再開になる。",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    try:
        del lmp._predictor
    except Exception:
        pass

    return summary


def main() -> None:
    args = build_parser().parse_args()
    config = WorkflowConfig.from_namespace(args)
    summary = run_workflow(config)
    if config.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
