from __future__ import annotations

"""CASTEP の 750 個前後の snapshot から txt(JSON) を出力し、
再読み込んで UMA の single-point 計算まで実行する runner。

主な処理:
1. CASTEP raw continuation md を連結して読む
2. 2.0 fs 間隔で 0.0〜1500.0 fs の snapshot を抽出する
3. 各 snapshot を 1 frame = 1 txt(JSON) に保存する
4. txt を再読み込んで round-trip を確認する
5. txt を入力にして uma-m-1p1.pt で single-point energy を計算する
6. DFT(E1) と UMA の parity を csv/json/png/md で出力する

このスクリプトは、構造抽出から単点計算と図の作成までを
1 本で実行することを目的とする。
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from static_energy_pipeline_audit import (
    ParsedFrame,
    build_dft_energy_table,
    build_parity_table,
    compute_parity_stats,
    compute_uma_single_point_batch,
    parse_castep_md,
    plot_parity,
    plot_residual_timeseries,
    sample_frames_by_time,
    save_json,
    save_table_csv,
    _array_close,
    _snapshot_signature,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS_INPUTS_PATH = REPO_ROOT / "analysis_code_bundle" / "analysis_inputs.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "uma-m-1p1.pt"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results_long_reference_prep_20260327"
    / "04_outputs"
    / "fresh_oc20_lammps_fix_external_nvt_15000step_20260404_static_energy_compare_1p1_txt751"
)
DEFAULT_INTERVAL_FS = 2.0
DEFAULT_END_FS = 1500.0
DEFAULT_ENERGY_KEY = "E1"
DEFAULT_TASK_NAME = "oc20"
DEFAULT_DEVICE = "cuda"
ENERGY_DEFINITIONS = {
    "E1": "potential energy",
    "E2": "Hamiltonian energy",
    "E3": "kinetic energy",
}


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="Export 2.0 fs CASTEP snapshots to txt, reload them, and run uma-m-1p1 single-point energy."
    )
    parser.add_argument("--config-path", "--manifest-path", dest="config_path", type=Path, default=DEFAULT_ANALYSIS_INPUTS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interval-fs", type=float, default=DEFAULT_INTERVAL_FS)
    parser.add_argument("--start-fs", type=float, default=0.0)
    parser.add_argument("--end-fs", type=float, default=DEFAULT_END_FS)
    parser.add_argument("--energy-key", choices=["E1", "E2", "E3"], default=DEFAULT_ENERGY_KEY)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Read existing 03/04 tables from output-dir and regenerate only the 05 figures without recalculating energies.",
    )
    return parser


def load_analysis_inputs(config_path: Path) -> dict[str, object]:
    """Read analysis inputs from the bundle JSON."""
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def make_snapshot_txt_name(frame: ParsedFrame, order_index: int) -> str:
    """snapshot 用 txt ファイル名を作る。

    frame index と time(fs) を含めて、順序と時刻が分かるようにする。
    """
    time_label = f"{frame.time_fs:07.1f}".replace("-", "m")
    return f"snapshot_{order_index:04d}_frame_{frame.global_frame:05d}_time_{time_label}fs.txt"


def parsed_frame_to_txt_payload(frame: ParsedFrame) -> dict[str, object]:
    """ParsedFrame を txt(JSON) の保存用 payload に変換する。"""
    return {
        "format_version": 1,
        "length_unit": "angstrom",
        "source_length_unit": "bohr",
        "frame_index": int(frame.global_frame),
        "local_frame": int(frame.local_frame),
        "source_index": int(frame.source_index),
        "source_file": frame.source_file,
        "source_path": frame.source_path,
        "raw_marker": float(frame.raw_marker),
        "time_fs": float(frame.time_fs),
        "cell_A": np.asarray(frame.cell_A, dtype=float).tolist(),
        "symbols": list(frame.symbols),
        "positions_A": np.asarray(frame.positions_A, dtype=float).tolist(),
        "energy_candidates_hartree": {
            key: float(value) for key, value in frame.energy_candidates_hartree.items()
        },
        "temperature_raw_au": float(frame.temperature_raw_au),
        "line_refs": frame.line_refs,
        "snapshot_signature": _snapshot_signature(frame.symbols, frame.cell_A, frame.positions_A),
    }


def save_sampled_snapshots_txt(
    sampled_frames: list[ParsedFrame],
    out_dir: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    """sampled snapshot を 1 frame = 1 txt として保存する。

    txt の中身は JSON とし、拡張子だけ `.txt` にする。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    txt_paths: list[Path] = []

    for order_index, frame in enumerate(sampled_frames):
        txt_path = out_dir / make_snapshot_txt_name(frame, order_index)
        payload = parsed_frame_to_txt_payload(frame)
        txt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        txt_paths.append(txt_path)
        manifest_rows.append(
            {
                "sample_order": order_index,
                "frame_index": frame.global_frame,
                "time_fs": frame.time_fs,
                "txt_path": str(txt_path),
                "source_file": frame.source_file,
                "local_frame": frame.local_frame,
            }
        )
    return manifest_rows, txt_paths


def load_snapshot_txt(path: Path) -> ParsedFrame:
    """txt(JSON) から ParsedFrame を復元する。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    length_unit = str(payload.get("length_unit", "bohr")).lower()
    cell = np.asarray(payload["cell_A"], dtype=float)
    positions = np.asarray(payload["positions_A"], dtype=float)
    if length_unit == "bohr":
        from ase.units import Bohr

        scale = float(Bohr)
        cell = cell * scale
        positions = positions * scale
    elif length_unit != "angstrom":
        raise ValueError(f"unsupported length_unit: {length_unit}")
    return ParsedFrame(
        global_frame=int(payload["frame_index"]),
        local_frame=int(payload["local_frame"]),
        source_index=int(payload["source_index"]),
        source_file=str(payload["source_file"]),
        source_path=str(payload["source_path"]),
        raw_marker=float(payload["raw_marker"]),
        time_fs=float(payload["time_fs"]),
        cell_A=cell,
        symbols=tuple(str(symbol) for symbol in payload["symbols"]),
        positions_A=positions,
        energy_candidates_hartree={
            str(key): float(value) for key, value in payload["energy_candidates_hartree"].items()
        },
        temperature_raw_au=float(payload["temperature_raw_au"]),
        line_refs=payload["line_refs"],
    )


def load_sampled_snapshots_txt(paths: Iterable[Path]) -> list[ParsedFrame]:
    """複数の txt を読み込んで ParsedFrame の一覧にする。"""
    return [load_snapshot_txt(path) for path in paths]


def verify_txt_roundtrip(
    original_frames: list[ParsedFrame],
    loaded_frames: list[ParsedFrame],
) -> dict[str, object]:
    """txt 化した snapshot が元データと一致するか確認する。"""
    if len(original_frames) != len(loaded_frames):
        raise AssertionError(
            f"roundtrip frame count mismatch: original={len(original_frames)} loaded={len(loaded_frames)}"
        )

    row_checks: list[dict[str, object]] = []
    for original, loaded in zip(original_frames, loaded_frames):
        row_checks.append(
            {
                "frame_index": original.global_frame,
                "time_fs": original.time_fs,
                "symbols_match": tuple(loaded.symbols) == tuple(original.symbols),
                "cell_match": _array_close(loaded.cell_A, original.cell_A, atol=1e-12),
                "positions_match": _array_close(loaded.positions_A, original.positions_A, atol=1e-12),
                "time_match": math.isclose(loaded.time_fs, original.time_fs, rel_tol=0.0, abs_tol=1e-12),
                "energy_E1_match": math.isclose(
                    loaded.energy_candidates_hartree["E1"],
                    original.energy_candidates_hartree["E1"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "energy_E2_match": math.isclose(
                    loaded.energy_candidates_hartree["E2"],
                    original.energy_candidates_hartree["E2"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "energy_E3_match": math.isclose(
                    loaded.energy_candidates_hartree["E3"],
                    original.energy_candidates_hartree["E3"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "signature_match": _snapshot_signature(
                    loaded.symbols, loaded.cell_A, loaded.positions_A
                )
                == _snapshot_signature(original.symbols, original.cell_A, original.positions_A),
            }
        )

    failures = [
        row for row in row_checks if not all(bool(value) for key, value in row.items() if key not in {"frame_index", "time_fs"})
    ]
    return {
        "snapshot_count": len(row_checks),
        "all_passed": len(failures) == 0,
        "failure_count": len(failures),
        "sample_failures": failures[:5],
        "row_checks_preview": row_checks[:10],
    }


def build_sign_bias_summary(merged_rows: list[dict[str, object]]) -> dict[str, object]:
    """DFT と UMA の相対エネルギーの符号傾向を集計する。

    正負の偏りと符号一致率を確認する。
    """
    dft = np.asarray([row["dft_energy_relative_eV_per_atom"] for row in merged_rows], dtype=float)
    uma = np.asarray([row["uma_energy_relative_eV_per_atom"] for row in merged_rows], dtype=float)

    def sign_label(values: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        labels = np.zeros(values.shape[0], dtype=int)
        labels[values > tol] = 1
        labels[values < -tol] = -1
        return labels

    dft_sign = sign_label(dft)
    uma_sign = sign_label(uma)
    total = int(dft.size)
    return {
        "n": total,
        "dft_positive_fraction": float(np.mean(dft_sign == 1)),
        "dft_negative_fraction": float(np.mean(dft_sign == -1)),
        "uma_positive_fraction": float(np.mean(uma_sign == 1)),
        "uma_negative_fraction": float(np.mean(uma_sign == -1)),
        "same_sign_fraction": float(np.mean(dft_sign == uma_sign)),
        "dft_positive_uma_negative_fraction": float(np.mean((dft_sign == 1) & (uma_sign == -1))),
        "dft_negative_uma_positive_fraction": float(np.mean((dft_sign == -1) & (uma_sign == 1))),
    }


def compute_window_summary(
    merged_rows: list[dict[str, object]],
    windows: list[tuple[float, float]],
) -> list[dict[str, object]]:
    """指定した時間窓ごとの parity 統計を集計する。"""
    summary_rows: list[dict[str, object]] = []
    for window_index, (start_fs, end_fs) in enumerate(windows):
        if window_index < len(windows) - 1:
            window_rows = [row for row in merged_rows if start_fs <= float(row["time_fs"]) < end_fs]
        else:
            window_rows = [row for row in merged_rows if start_fs <= float(row["time_fs"]) <= end_fs]
        if len(window_rows) < 2:
            continue
        stats = compute_parity_stats(window_rows)
        residual = np.asarray([row["residual_eV_per_atom"] for row in window_rows], dtype=float)
        summary_rows.append(
            {
                "window_start_fs": start_fs,
                "window_end_fs": end_fs,
                "n": len(window_rows),
                "rho": float(stats["rho"]),
                "R2": float(stats["R2"]),
                "RMSE_eV": float(stats["RMSE_eV"]),
                "slope": float(stats["slope"]),
                "residual_mean_eV_per_atom": float(np.mean(residual)),
                "residual_std_eV_per_atom": float(np.std(residual)),
            }
        )
    return summary_rows


def write_markdown_report(
    out_path: Path,
    sample_count: int,
    interval_fs: float,
    roundtrip_summary: dict[str, object],
    stats: dict[str, object],
    sign_summary: dict[str, object],
    window_summary: list[dict[str, object]],
    txt_manifest_path: Path,
    parity_csv_path: Path,
) -> None:
    """Write a markdown summary of the pipeline output."""
    lines: list[str] = []
    lines.append("# 1p1 analysis for about 750 txt snapshots")
    lines.append("")
    lines.append("## Overview")
    lines.append("- Extract CASTEP raw continuation at 2.0 fs intervals, save txt(JSON), run UMA single-point, and compare parity.")
    lines.append("- Compare by physical time `t`, not by step id.")
    lines.append("")
    lines.append("## Basic info")
    lines.append(f"- sample_count: {sample_count}")
    lines.append(f"- sample_interval_fs: {interval_fs}")
    lines.append("- DFT side: E1 = potential energy")
    lines.append("- Energy is reported as eV/atom.")
    lines.append("- UMA checkpoint: uma-m-1p1.pt")
    lines.append(f"- txt manifest: `{txt_manifest_path}`")
    lines.append(f"- parity csv: `{parity_csv_path}`")
    lines.append("")
    lines.append("## txt round-trip")
    lines.append(f"- all_passed: {roundtrip_summary['all_passed']}")
    lines.append(f"- failure_count: {roundtrip_summary['failure_count']}")
    lines.append("")
    lines.append("## parity stats")
    lines.append(f"- rho: {stats['rho']:.6f}")
    lines.append(f"- R2: {stats['R2']:.6f}")
    lines.append(f"- RMSE_eV_per_atom: {stats['RMSE_eV']:.6f}")
    lines.append(f"- slope: {stats['slope']:.6f}")
    lines.append(f"- residual_mean_eV_per_atom: {stats['residual_mean_eV']:.6f}")
    lines.append("")
    lines.append("## sign bias")
    lines.append(f"- dft_positive_fraction: {sign_summary['dft_positive_fraction']:.6f}")
    lines.append(f"- uma_negative_fraction: {sign_summary['uma_negative_fraction']:.6f}")
    lines.append(f"- same_sign_fraction: {sign_summary['same_sign_fraction']:.6f}")
    lines.append(f"- dft_positive_uma_negative_fraction: {sign_summary['dft_positive_uma_negative_fraction']:.6f}")
    lines.append("")
    lines.append("## window summary")
    lines.append("| start_fs | end_fs | n | rho | slope | RMSE_eV/atom | residual_mean_eV/atom |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in window_summary:
        lines.append(
            f"| {row['window_start_fs']:.1f} | {row['window_end_fs']:.1f} | {row['n']} | {row['rho']:.4f} | {row['slope']:.4f} | {row['RMSE_eV']:.4f} | {row['residual_mean_eV_per_atom']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- The txt body is JSON, but the extension remains `.txt`.")
    lines.append("- The DFT/UMA comparison uses physical time `t`, not step id.")
    lines.append("- Relative energy mainly uses E1 and is normalized per atom; raw total energies are kept for reference.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV を dict 行列として読む。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def regenerate_figures_from_existing_tables(output_dir: Path) -> dict[str, object]:
    """既存の 03/04 テーブルから eV/atom 図だけを作り直す。"""
    dft_csv = output_dir / "03_dft_energy_table.csv"
    uma_csv = output_dir / "04_uma_energy_table.csv"
    if not dft_csv.exists():
        raise FileNotFoundError(f"missing DFT table: {dft_csv}")
    if not uma_csv.exists():
        raise FileNotFoundError(f"missing UMA table: {uma_csv}")

    dft_rows = load_csv_rows(dft_csv)
    uma_rows = load_csv_rows(uma_csv)
    merged_rows, merged_meta = build_parity_table(dft_rows, uma_rows, zero_mode="t0")
    stats = compute_parity_stats(merged_rows)

    parity_csv_path = output_dir / "05_parity_table.csv"
    parity_png_path = output_dir / "05_parity.png"
    residual_png_path = output_dir / "05_residual_timeseries.png"
    parity_meta_path = output_dir / "05_parity_meta.json"
    parity_stats_path = output_dir / "05_parity_stats.json"

    save_table_csv(parity_csv_path, merged_rows)
    plot_parity(merged_rows, stats, parity_png_path)
    plot_residual_timeseries(merged_rows, residual_png_path)
    save_json(parity_meta_path, merged_meta)
    save_json(parity_stats_path, stats)

    return {
        "source_dft_csv": str(dft_csv),
        "source_uma_csv": str(uma_csv),
        "parity_csv": str(parity_csv_path),
        "parity_png": str(parity_png_path),
        "residual_png": str(residual_png_path),
        "row_count": len(merged_rows),
        "rho": float(stats["rho"]),
        "R2": float(stats["R2"]),
        "RMSE_eV_per_atom": float(stats["RMSE_eV"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = args.output_dir / "txt_snapshots"

    if args.replot_only:
        summary = regenerate_figures_from_existing_tables(args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # 1. load bundle JSON
    config = load_analysis_inputs(args.config_path)
    base_castep_path = REPO_ROOT / str(config["base_condition_file"])
    continuation_paths = [REPO_ROOT / rel for rel in config["trajectory_segments"]]
    duplicate_tolerance_A = float(config["boundary_duplicate_tolerance_angstrom"])

    # 2. read only the raw CASTEP continuation needed for sampling
    frames, parse_meta = parse_castep_md(
        base_castep_path=base_castep_path,
        continuation_paths=continuation_paths,
        duplicate_tolerance_A=duplicate_tolerance_A,
    )

    # 3. sample snapshots every 2.0 fs from 0.0 to 1500.0 fs
    sampled_frames, selection_rows = sample_frames_by_time(
        frames=frames,
        interval_fs=float(args.interval_fs),
        start_fs=float(args.start_fs),
        end_fs=float(args.end_fs),
    )
    if len(sampled_frames) < 700:
        raise AssertionError(f"Expected about 750 snapshots, but got only {len(sampled_frames)}")

    # 4. save sampled snapshots and manifests
    sampling_csv_path = args.output_dir / "00_sampling_manifest.csv"
    sampling_json_path = args.output_dir / "00_sampling_manifest.json"
    save_table_csv(sampling_csv_path, selection_rows)
    save_json(
        sampling_json_path,
        {
            "interval_fs": float(args.interval_fs),
            "start_fs": float(args.start_fs),
            "end_fs": float(args.end_fs),
            "sample_count": len(sampled_frames),
            "selection_rows": selection_rows,
            "parse_meta": parse_meta,
        },
    )

    txt_manifest_rows, txt_paths = save_sampled_snapshots_txt(sampled_frames, txt_dir)
    txt_manifest_csv_path = args.output_dir / "01_txt_snapshot_manifest.csv"
    txt_manifest_json_path = args.output_dir / "01_txt_snapshot_manifest.json"
    save_table_csv(txt_manifest_csv_path, txt_manifest_rows)
    save_json(
        txt_manifest_json_path,
        {
            "txt_snapshot_count": len(txt_paths),
            "txt_paths": [str(path) for path in txt_paths],
        },
    )

    # 5. verify txt round-trip
    loaded_frames = load_sampled_snapshots_txt(txt_paths)
    roundtrip_summary = verify_txt_roundtrip(sampled_frames, loaded_frames)
    if not roundtrip_summary["all_passed"]:
        raise AssertionError(f"txt round-trip failed: {roundtrip_summary['sample_failures']}")
    save_json(args.output_dir / "02_txt_roundtrip_check.json", roundtrip_summary)

    # 6. convert DFT(E1) to relative energy
    dft_rows, dft_meta = build_dft_energy_table(
        sampled_frames=loaded_frames,
        energy_key=args.energy_key,
        zero_mode="t0",
    )
    save_table_csv(args.output_dir / "03_dft_energy_table.csv", dft_rows)
    save_json(args.output_dir / "03_dft_energy_table.json", dft_meta | {"row_count": len(dft_rows)})

    # 7. run UMA single-point energy on txt snapshots
    uma_rows, uma_meta, _calculator = compute_uma_single_point_batch(
        sampled_frames=loaded_frames,
        model_path=args.model_path,
        task_name=args.task_name,
        device=args.device,
    )
    save_table_csv(args.output_dir / "04_uma_energy_table.csv", uma_rows)
    save_json(args.output_dir / "04_uma_energy_table.json", uma_meta | {"row_count": len(uma_rows)})

    # 8. compare DFT and UMA parity
    merged_rows, merged_meta = build_parity_table(dft_rows, uma_rows, zero_mode="t0")
    stats = compute_parity_stats(merged_rows)
    sign_summary = build_sign_bias_summary(merged_rows)
    window_summary = compute_window_summary(
        merged_rows,
        windows=[(0.0, 100.0), (100.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0)],
    )

    parity_csv_path = args.output_dir / "05_parity_table.csv"
    parity_meta_path = args.output_dir / "05_parity_meta.json"
    parity_stats_path = args.output_dir / "05_parity_stats.json"
    sign_summary_path = args.output_dir / "05_sign_bias_summary.json"
    window_summary_csv_path = args.output_dir / "05_window_summary.csv"
    parity_png_path = args.output_dir / "05_parity.png"
    residual_png_path = args.output_dir / "05_residual_timeseries.png"
    report_md_path = args.output_dir / "06_report_txt751.md"

    save_table_csv(parity_csv_path, merged_rows)
    save_json(parity_meta_path, merged_meta)
    save_json(parity_stats_path, stats)
    save_json(sign_summary_path, sign_summary)
    save_table_csv(window_summary_csv_path, window_summary)
    plot_parity(merged_rows, stats, parity_png_path)
    plot_residual_timeseries(merged_rows, residual_png_path)
    write_markdown_report(
        out_path=report_md_path,
        sample_count=len(sampled_frames),
        interval_fs=float(args.interval_fs),
        roundtrip_summary=roundtrip_summary,
        stats=stats,
        sign_summary=sign_summary,
        window_summary=window_summary,
        txt_manifest_path=txt_manifest_csv_path,
        parity_csv_path=parity_csv_path,
    )

    # 9. write summary output
    summary = {
        "sample_count": len(sampled_frames),
        "interval_fs": float(args.interval_fs),
        "time_range_fs": [float(args.start_fs), float(args.end_fs)],
        "txt_roundtrip_all_passed": roundtrip_summary["all_passed"],
        "rho": float(stats["rho"]),
        "R2": float(stats["R2"]),
        "RMSE_eV": float(stats["RMSE_eV"]),
        "slope": float(stats["slope"]),
        "residual_mean_eV": float(stats["residual_mean_eV"]),
        "dft_positive_uma_negative_fraction": float(sign_summary["dft_positive_uma_negative_fraction"]),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
