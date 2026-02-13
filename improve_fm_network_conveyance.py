#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is script that helps to detect problems in the cross-section conveyance, 
- where the curve has negative spikes 
- and optionally improve cross-section conveyance for Flood Modeller .DAT file

Pipeline:
 1) Detect dips (ACTIVE conveyance)
 2) Nearest‑by‑Y: for each dip stage, try equally‑nearest points one by one; keep only those
    that remove the targeted dip; stop for that dip once removed.
 3) Re‑check
 4) Bank‑aware: for remaining dips, try bank candidates; keep only if they remove the targeted dip.
 5) Re‑check
 6) Final nearest‑by‑Y: for any remaining dips, again try equally‑nearest points; keep only those
    that remove the targeted dip.
 7) Plot remaining dips with cross‑section shape in background + conveyance curves.

Requirements:
 - floodmodeller_api (pip install floodmodeller-api)
 - numpy, pandas, matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional

# Flood Modeller API
from floodmodeller_api import DAT
from floodmodeller_api.units.conveyance import calculate_cross_section_conveyance

# ------------------------
# Utilities and constants
# ------------------------
INVALID_WIN_CHARS = r'[\\\\<>:"/\\n\\?\\*]'
CONFIG = {
    "NEG_DROP_REL": 0.01,
    "MIN_POINTS": 6,
    "PLOT_MAX_SECTIONS": 50,
}

def safe_filename(name: str) -> str:
    sanitized = re.sub(INVALID_WIN_CHARS, "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip(" .")
    return sanitized

# ------------------------
# Dip detection (ACTIVE conveyance)
# ------------------------

def detect_dips(stage: np.ndarray, K: np.ndarray, rel_drop: float, minK: float, cfg=CONFIG):
    sort_idx = np.argsort(stage)
    z = stage[sort_idx]
    k = K[sort_idx]
    if len(z) < cfg["MIN_POINTS"]:
        return {"dips_idx": [], "dips_z": []}
    dK = np.diff(k)
    prevK = k[:-1]
    epsilon = 1e-9
    rel_cond = dK < -(rel_drop * np.maximum(prevK, epsilon))
    floor_cond = prevK >= minK
    dips_mask = rel_cond & floor_cond
    dips_idx = list(np.where(dips_mask)[0])
    dips_z = [z[i+1] for i in dips_idx]
    return {"dips_idx": dips_idx, "dips_z": dips_z}

# ------------------------
# Nearest-by-elevation placement helpers
# ------------------------

def _candidate_indices_by_Y(df: pd.DataFrame, stage_z: float, y_tol: Optional[float] = None,
                            avoid_edges: bool = True, tie_atol: float = 1e-6) -> List[int]:
    y = df["Y"].values
    if y.size == 0:
        return []
    idx_all = np.arange(len(y))
    if avoid_edges and len(y) >= 3:
        idx_all = idx_all[1:-1]
    deltas = np.abs(y[idx_all] - stage_z)
    if y_tol is not None:
        mask_tol = deltas <= y_tol
        idx_all = idx_all[mask_tol]
        deltas = deltas[mask_tol]
        if deltas.size == 0:
            return []
    order = np.argsort(deltas)
    idx_sorted = idx_all[order]
    del_sorted = deltas[order]
    min_delta = del_sorted[0]
    tie_mask = np.isclose(del_sorted, min_delta, rtol=0.0, atol=tie_atol)
    candidates = [int(idx_sorted[i]) for i in range(len(idx_sorted)) if tie_mask[i]]
    return candidates


def _set_panel(df: pd.DataFrame, idx: int) -> bool:
    if "Panel" not in df.columns:
        df["Panel"] = False
    if bool(df.loc[df.index[idx], "Panel"]) is True:
        return False
    df.loc[df.index[idx], "Panel"] = True
    return True

# ------------------------
# Bank-aware inference (heuristic)
# ------------------------

def _central_diff(y, x):
    slope = np.full_like(y, np.nan, dtype=float)
    if len(y) >= 3:
        denom = (x[2:] - x[:-2])
        denom[denom == 0] = np.nan
        slope[1:-1] = (y[2:] - y[:-2]) / denom
    return slope


def _curvature_from_slope(slope):
    curv = np.full_like(slope, np.nan, dtype=float)
    if len(slope) >= 3:
        curv[1:-1] = slope[2:] - slope[:-2]
    return curv


def _find_thalweg_indices(y, window=2):
    i_min = int(np.argmin(y))
    lo = max(i_min - window, 0)
    hi = min(i_min + window, len(y) - 1)
    return np.arange(lo, hi + 1)


def _bank_candidates(df: pd.DataFrame,
                     slope_thr: float = 0.05,
                     curv_thr: float = 0.05,
                     edge_margin: int = 2,
                     attr_window: int = 3) -> Tuple[Optional[int], Optional[int]]:
    x = df["X"].values.astype(float)
    y = df["Y"].values.astype(float)
    n = df["Mannings n"].values if "Mannings n" in df.columns else None
    rpl = df["RPL"].values if "RPL" in df.columns else None
    if len(x) < 6:
        return None, None
    slope = _central_diff(y, x)
    curv = _curvature_from_slope(slope)
    thalweg_ix = _find_thalweg_indices(y, window=2)
    i_center = int(np.median(thalweg_ix))
    i_left = None
    for i in range(i_center, edge_margin, -1):
        ds = 0.0
        if np.isfinite(slope[i]) and np.isfinite(slope[i-1]):
            ds = slope[i] - slope[i-1]
        if abs(ds) >= slope_thr or (np.isfinite(curv[i]) and abs(curv[i]) >= curv_thr):
            i_left = i
            break
    i_right = None
    for i in range(i_center, len(x) - edge_margin - 1):
        ds = 0.0
        if np.isfinite(slope[i+1]) and np.isfinite(slope[i]):
            ds = slope[i+1] - slope[i]
        if abs(ds) >= slope_thr or (np.isfinite(curv[i]) and abs(curv[i]) >= curv_thr):
            i_right = i
            break
    def refine(i_candidate):
        if i_candidate is None:
            return None
        lo = max(i_candidate - attr_window, edge_margin)
        hi = min(i_candidate + attr_window, len(x) - edge_margin - 1)
        best = i_candidate
        if n is not None and len(n) == len(y):
            for j in range(lo, hi+1):
                if j+1 < len(n) and abs(float(n[j+1]) - float(n[j])) > 1e-6:
                    best = j+1
                    break
        if rpl is not None and len(rpl) == len(y):
            for j in range(lo, hi+1):
                if j+1 < len(rpl) and abs(float(rpl[j+1]) - float(rpl[j])) > 1e-9:
                    best = j+1
                    break
        return best
    i_left = refine(i_left)
    i_right = refine(i_right)
    def valid(i):
        if i is None:
            return False
        if i <= edge_margin or i >= len(x) - edge_margin - 1:
            return False
        return y[i] >= y[i_center]
    i_left = i_left if valid(i_left) else None
    i_right = i_right if valid(i_right) else None
    return i_left, i_right

# ------------------------
# ACTIVE conveyance helper
# ------------------------

def _compute_active_conveyance(section) -> pd.Series:
    return calculate_cross_section_conveyance(
        x=section.active_data.X.values,
        y=section.active_data.Y.values,
        n=section.active_data["Mannings n"].values,
        rpl=section.active_data.RPL.values,
        panel_markers=section.active_data.Panel.values,
    )

# ------------------------
# Improvement test helpers
# ------------------------

def _dip_present_near(z_target: float, dips_z: List[float], z_tol: float) -> bool:
    return any(abs(float(d) - float(z_target)) <= z_tol for d in dips_z)


def _try_apply_panel_if_improves(section, idx: int, z_target: float,
                                 rel_drop: float, minK: float, z_tol: float) -> bool:
    """Set Panel=True at idx, recompute ACTIVE conveyance, and keep it only if the
    targeted dip at z_target is no longer detected within z_tol. Otherwise revert.
    Returns True if change is kept, False if reverted.
    """
    df = section.data
    if "Panel" not in df.columns:
        df["Panel"] = False
    # Already a panel? then there's nothing to improve here
    if bool(df.loc[df.index[idx], "Panel"]) is True:
        return False
    # Apply candidate
    df.loc[df.index[idx], "Panel"] = True
    try:
        conv = _compute_active_conveyance(section)
        stage = np.asarray(conv.index, dtype=float)
        K = np.asarray(conv.values, dtype=float)
        res = detect_dips(stage, K, rel_drop, minK, CONFIG)
        improved = not _dip_present_near(z_target, res["dips_z"], z_tol)
    except Exception:
        improved = False
    if not improved:
        # revert
        df.loc[df.index[idx], "Panel"] = False
        return False
    return True

# ------------------------
# Plot helper: cross-section shape + conveyance twin axis
# ------------------------

def _plot_section_with_remaining_dips(section, conv_series: pd.Series, dips_res: dict, out_png: Path):
    df = section.data
    X = df["X"].values
    Y = df["Y"].values

    stage = np.asarray(conv_series.index, dtype=float)
    K = np.asarray(conv_series.values, dtype=float)
    sort_idx = np.argsort(stage)
    z_sorted = stage[sort_idx]
    k_sorted = K[sort_idx]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(X, Y, color="saddlebrown", lw=1.6)
    ax1.fill_between(X, Y, np.nanmin(Y) - 0.1, color="saddlebrown", alpha=0.35)
    ax1.set_xlabel("Chainage (m)", color="saddlebrown")
    ax1.set_ylabel("Stage (m)")

    ax2 = ax1.twiny()
    ax2.plot(k_sorted, z_sorted, "b-", lw=1.8, label="Conveyance (active)")

    dips_idx_sorted = np.array(dips_res.get("dips_idx", []))
    if dips_idx_sorted.size > 0:
        ax2.scatter(k_sorted[:-1][dips_idx_sorted], z_sorted[:-1][dips_idx_sorted],
                    color="red", s=34, zorder=5, label="Remaining dip")

    ax2.set_xlabel("Conveyance K (m³/s)", color="b")
    ax1.grid(True, alpha=0.25)
    handles = []
    labels = []
    h1, = ax1.plot([], [], color="saddlebrown", lw=1.6)
    handles.append(h1); labels.append("Cross-section shape")
    h2, = ax2.plot([], [], "b-", lw=1.8)
    handles.append(h2); labels.append("Conveyance (active)")
    if dips_idx_sorted.size > 0:
        h3 = ax2.scatter([], [], color="red", s=34)
        handles.append(h3); labels.append("Remaining dip")
    ax2.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ------------------------
# Main workflow
# ------------------------

def analyze_dat(dat_path: Path,
                rel_drop: float,
                minK: float,
                csv_out: Optional[Path] = None,
                plots_dir: Optional[Path] = None,
                apply_panels: bool = False,
                y_tol: Optional[float] = None,
                output_dat: Optional[Path] = None,
                slope_thr: float = 0.05,
                curv_thr: float = 0.05,
                min_spacing: int = 2):
    dat = DAT(str(dat_path))
    sections_dict = dat.sections
    flagged = []
    save_plots = plots_dir is not None
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    # Internal tolerance for matching the targeted dip stage
    z_tol = float(y_tol) if y_tol is not None else 0.05  # m

    # ---- Pass 1: detect dips (ACTIVE) ----
    for name, section in sections_dict.items():
        if not hasattr(section, "active_data"):
            continue
        try:
            active_conv = _compute_active_conveyance(section)
            stage = np.asarray(active_conv.index, dtype=float)
            K = np.asarray(active_conv.values, dtype=float)
        except Exception:
            continue
        res = detect_dips(stage, K, rel_drop, minK, CONFIG)
        num_dips = len(res["dips_idx"])
        if num_dips > 0:
            stages_flagged = sorted(list(set(res["dips_z"])) )
            flagged.append({"name": name, "num_dips": num_dips, "stages": stages_flagged})
            for z in res["dips_z"]:
                csv_rows.append({"section": name, "type": "dip_before", "stage_m": z})

    # ---- Pass 2: nearest-by-Y (try candidates; keep only if targeted dip removed) ----
    applied_nearest1 = []
    remaining_stages_map = {}
    if apply_panels and flagged:
        for item in flagged:
            name = item["name"]
            section = sections_dict[name]
            df = section.data
            if "Panel" not in df.columns:
                df["Panel"] = False
            changed_first = []
            for z in sorted(set(item["stages"])):
                candidates = _candidate_indices_by_Y(df, z, y_tol=y_tol)
                if not candidates:
                    continue
                # Try each candidate; accept the first that removes the targeted dip
                applied_for_this_dip = False
                for i0 in candidates:
                    kept = _try_apply_panel_if_improves(section, i0, z, rel_drop, minK, z_tol)
                    if kept:
                        changed_first.append(i0)
                        applied_for_this_dip = True
                        break  # stop once the dip at z is removed
                # If none helped, no change for this dip
            if changed_first:
                applied_nearest1.append({"name": name, "indices": sorted(set(changed_first))})
            # Recompute ACTIVE conveyance after nearest pass
            try:
                conv_after1 = _compute_active_conveyance(section)
                stage1 = np.asarray(conv_after1.index, dtype=float)
                K1 = np.asarray(conv_after1.values, dtype=float)
            except Exception:
                stage1, K1 = None, None
            remaining_stages = []
            if stage1 is not None:
                res1 = detect_dips(stage1, K1, rel_drop, minK, CONFIG)
                remaining_stages = sorted(list(set(res1["dips_z"])) )
                for z in res1["dips_z"]:
                    csv_rows.append({"section": name, "type": "dip_after_nearest1", "stage_m": z})
            remaining_stages_map[name] = remaining_stages

    # ---- Pass 3: bank-aware (try candidates; keep only if targeted dip removed) ----
    applied_bankaware = []
    remaining_after_bank_map = {}
    if apply_panels and flagged:
        for item in flagged:
            name = item["name"]
            section = sections_dict[name]
            rem = remaining_stages_map.get(name, [])
            if not rem:
                remaining_after_bank_map[name] = []
                continue
            changed_idx = []
            df = section.data
            if "Panel" not in df.columns:
                df["Panel"] = False
            i_left, i_right = _bank_candidates(df, slope_thr=slope_thr, curv_thr=curv_thr)
            y = df["Y"].values
            # For each remaining dip stage, sort bank candidates by proximity in elevation
            for z in sorted(set(rem)):
                bank_candidates = []
                if i_left is not None:
                    bank_candidates.append((abs(y[i_left] - z), i_left))
                if i_right is not None:
                    bank_candidates.append((abs(y[i_right] - z), i_right))
                bank_candidates.sort(key=lambda t: t[0])
                applied_for_this_dip = False
                for _, idx in bank_candidates:
                    kept = _try_apply_panel_if_improves(section, idx, z, rel_drop, minK, z_tol)
                    if kept:
                        changed_idx.append(idx)
                        applied_for_this_dip = True
                        break
                # If none helped, leave as is; final nearest pass will try again
            if changed_idx:
                applied_bankaware.append({"name": name, "indices": changed_idx})
            # Recompute ACTIVE conveyance after bank-aware pass
            try:
                conv_after2 = _compute_active_conveyance(section)
                stage2 = np.asarray(conv_after2.index, dtype=float)
                K2 = np.asarray(conv_after2.values, dtype=float)
            except Exception:
                stage2, K2 = None, None
            remaining_after_bank = []
            if stage2 is not None:
                res2 = detect_dips(stage2, K2, rel_drop, minK, CONFIG)
                remaining_after_bank = sorted(list(set(res2["dips_z"])) )
                for z in res2["dips_z"]:
                    csv_rows.append({"section": name, "type": "dip_after_bankaware", "stage_m": z})
            remaining_after_bank_map[name] = remaining_after_bank

    # ---- Pass 4: final nearest-by-Y (try candidates; keep only if targeted dip removed) ----
    applied_nearest2 = []
    if apply_panels and flagged:
        for item in flagged:
            name = item["name"]
            section = sections_dict[name]
            rem2 = remaining_after_bank_map.get(name, [])
            if not rem2:
                continue
            df = section.data
            if "Panel" not in df.columns:
                df["Panel"] = False
            changed_final = []
            for z in sorted(set(rem2)):
                candidates = _candidate_indices_by_Y(df, z, y_tol=y_tol)
                if not candidates:
                    continue
                applied_for_this_dip = False
                for i0 in candidates:
                    kept = _try_apply_panel_if_improves(section, i0, z, rel_drop, minK, z_tol)
                    if kept:
                        changed_final.append(i0)
                        applied_for_this_dip = True
                        break
            if changed_final:
                applied_nearest2.append({"name": name, "indices": sorted(set(changed_final))})
            # Recompute ACTIVE conveyance after final nearest pass
            try:
                conv_after3 = _compute_active_conveyance(section)
                stage3 = np.asarray(conv_after3.index, dtype=float)
                K3 = np.asarray(conv_after3.values, dtype=float)
            except Exception:
                stage3, K3 = None, None
            if stage3 is not None:
                res3 = detect_dips(stage3, K3, rel_drop, minK, CONFIG)
                for z in res3["dips_z"]:
                    csv_rows.append({"section": name, "type": "dip_after_nearest2", "stage_m": z})

    # ---- Final plots: sections with remaining kinks after all passes ----
    if save_plots:
        for name, section in sections_dict.items():
            try:
                conv_final = _compute_active_conveyance(section)
                stageF = np.asarray(conv_final.index, dtype=float)
                KF = np.asarray(conv_final.values, dtype=float)
            except Exception:
                continue
            resF = detect_dips(stageF, KF, rel_drop, minK, CONFIG)
            if len(resF["dips_idx"]) == 0:
                continue
            fname = safe_filename(name)
            out_png = plots_dir / f"{fname}_conveyance_dips_final.png"
            _plot_section_with_remaining_dips(section, conv_final, resF, out_png)

    # ---- CSV output ----
    if csv_out is not None:
        if flagged or applied_nearest1 or applied_bankaware or applied_nearest2:
            pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
        else:
            pd.DataFrame(columns=["section", "type", "stage_m"]).to_csv(csv_out, index=False)

    # ---- Console summary ----
    if flagged:
        print(f"\nSections with non‑monotonic conveyance dips (>{rel_drop*100:.2f}% drop; K_prev ≥ {minK:g}):")
        for item in flagged:
            stage_str = ", ".join(f"{z:.3f}" for z in item["stages"])
            print(f" - {item['name']}: {item['num_dips']} dip(s) at stage(s) [{stage_str}]")
    else:
        print(f"No sections with >{rel_drop*100:.2f}% conveyance dips detected (K_prev ≥ {minK:g}).")

    if apply_panels:
        if applied_nearest1:
            print("\nApplied panel markers (nearest‑by‑elevation; pass 1; only if targeted dip removed):")
            for s in applied_nearest1:
                idxs = ", ".join(str(i) for i in s["indices"]) 
                print(f" - {s['name']}: Panel=True at indices [{idxs}] (nearest pass 1)")
        if applied_bankaware:
            print("\nApplied panel markers (bank‑aware; pass 2; only if targeted dip removed):")
            for s in applied_bankaware:
                idxs = ", ".join(str(i) for i in s["indices"]) 
                print(f" - {s['name']}: Panel=True at indices [{idxs}] (bank‑aware pass)")
        if applied_nearest2:
            print("\nApplied panel markers (nearest‑by‑elevation; pass 3; only if targeted dip removed):")
            for s in applied_nearest2:
                idxs = ", ".join(str(i) for i in s["indices"]) 
                print(f" - {s['name']}: Panel=True at indices [{idxs}] (final nearest pass)")
        if output_dat is not None:
            out_path = Path(output_dat)
            dat.save(out_path)
            print(f"\nSaved improved network to: {out_path}")
        else:
            print("\n(Preview mode) No DAT saved. Use --output-dat to write changes, e.g. --output-dat EX3_improved.dat")

    return flagged, applied_nearest1, applied_bankaware, applied_nearest2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect dips (ACTIVE); nearest panels and bank‑aware panels are applied only if they remove the targeted dip; "
            "final nearest pass does the same; then save/plot remaining dips with cross‑section shapes. (CLI unchanged)"
        )
    )
    parser.add_argument("dat_file", type=str, help="Path to network DAT file (e.g., EX3.dat)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save CSV summary (dip points before/after).")
    parser.add_argument("--plots", type=str, default=None, help="Optional folder to save diagnostic plots (final remaining).")
    parser.add_argument("--rel", type=float, default=CONFIG["NEG_DROP_REL"],
                        help="Relative drop threshold (fraction). Example: 0.01 for 1%.")
    parser.add_argument("--minK", type=float, default=0.0,
                        help="Absolute conveyance floor. Only flag dips when previous K ≥ minK.")
    parser.add_argument("--apply-panels", action="store_true",
                        help="Apply nearest → bank‑aware → final nearest (each kept only if targeted dip is removed).")
    parser.add_argument("--y-tol", type=float, default=None,
                        help="Vertical tolerance (m) for snapping by elevation (also used as dip stage matching tolerance if provided).")
    parser.add_argument("--output-dat", type=str, default=None,
                        help="Path to save a copy of the DAT with applied panel markers.")
    parser.add_argument("--slope-thr", type=float, default=0.05,
                        help="Slope change threshold for bank detection.")
    parser.add_argument("--curv-thr", type=float, default=0.05,
                        help="Curvature threshold for bank detection.")
    parser.add_argument("--min-spacing", type=int, default=2,
                        help="Minimum index spacing between newly added panels.")
    args = parser.parse_args()

    if not (0.0 < args.rel < 1.0):
        raise ValueError(f"--rel must be a fraction between 0 and 1 (exclusive). Got: {args.rel}")
    if args.minK < 0.0:
        raise ValueError(f"--minK must be non‑negative. Got: {args.minK}")
    dat_path = Path(args.dat_file)
    if not dat_path.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_path.resolve()}")
    csv_out = Path(args.csv) if args.csv else None
    plots_dir = Path(args.plots) if args.plots else None

    analyze_dat(dat_path, rel_drop=args.rel, minK=args.minK,
                csv_out=csv_out, plots_dir=plots_dir,
                apply_panels=args.apply_panels, y_tol=args.y_tol,
                output_dat=Path(args.output_dat) if args.output_dat else None,
                slope_thr=args.slope_thr, curv_thr=args.curv_thr,
                min_spacing=args.min_spacing)


if __name__ == "__main__":
    main()


'''

1. Place the python script at the location where the network file is located
2. See Run Options below
3. Replace the name of the network file in the commands 'network.dat with network name you need to analyse


Run options:

# Detect only (CSV + plots of remaining kinks):
python improve_fm_network_conveyance.py network.dat --rel 0.01 --minK 1.0 --csv out_kinks.csv --plots plots_before



# Full pipeline (detect → improve → recheck → improve) and save the improved DAT:
python improve_fm_network_conveyance.py network.dat --apply-panels --output-dat network_improved.dat --rel 0.01 --minK 1.0 --csv out_kinks.csv --plots plots_after --y-tol 0.25 --slope-thr 0.05 --curv-thr 0.05 --min-spacing 0


'''
