"""
CrystalNet — Training Orchestrator
===================================
Runs all four training pipelines in sequence, collects metrics,
saves a summary text file, and produces training plots.

Usage:
    python run.py [--epochs N] [--folds K] [--targets ph salt peg temp]

Outputs:
    results/training_summary.txt  — human-readable metrics report
    results/metrics.json          — machine-readable JSON
    results/plots/                — PNG plots per target + comparison
"""

import sys
import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_TARGETS = ["ph", "salt", "peg", "temp"]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _read_config(path: Path) -> dict:
    """Load *_config.json saved by each train.py."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _collect_metrics() -> dict:
    metrics = {}
    metrics["ph"]   = _read_config(ROOT / "ph"   / "models" / "ph_config.json")
    metrics["salt"] = _read_config(ROOT / "salt"  / "models" / "salt_config.json")
    metrics["peg"]  = _read_config(ROOT / "peg"   / "models" / "peg_config.json")
    metrics["temp"] = _read_config(ROOT / "temp"  / "models" / "temp_config.json")
    return metrics


# ── Summary report ─────────────────────────────────────────────────────────────

def _write_summary(metrics: dict, timing: dict, args):
    lines = [
        "=" * 64,
        "CrystalNet Training Summary",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Epochs: {args.epochs}   Folds: {args.folds}",
        "=" * 64,
        "",
    ]

    # pH
    ph = metrics.get("ph", {})
    lines += [
        "── pH Prediction (regression) ─────────────────────────────",
        f"  OOF MAE  : {ph.get('oof_mae',  'N/A')}",
        f"  OOF RMSE : {ph.get('oof_rmse', 'N/A')}",
        f"  OOF R²   : {ph.get('oof_r2',   'N/A')}",
        f"  Meta coef: DL={ph.get('meta_coef',['?','?','?'])[0]:.3f}  "
        f"XGB={ph.get('meta_coef',['?','?','?'])[1]:.3f}  "
        f"LGB={ph.get('meta_coef',['?','?','?'])[2]:.3f}"
        if ph.get("meta_coef") else "  Meta coef: N/A",
        f"  Wall time : {timing.get('ph', 0):.1f}s",
        "",
    ]

    # Salt
    salt = metrics.get("salt", {})
    lines += [
        "── Salt Prediction (type clf + conc reg) ──────────────────",
        f"  Type Accuracy  : {salt.get('type_accuracy',  'N/A')}",
        f"  Type F1 (macro): {salt.get('type_f1',        'N/A')}",
        f"  Conc OOF MAE   : {salt.get('conc_oof_mae',   'N/A')}",
        f"  Conc OOF R²    : {salt.get('conc_oof_r2',    'N/A')}",
        f"  Wall time      : {timing.get('salt', 0):.1f}s",
        "",
    ]

    # PEG
    peg = metrics.get("peg", {})
    lines += [
        "── PEG Type Classification ─────────────────────────────────",
        f"  OOF Accuracy: {peg.get('oof_accuracy', 'N/A')}",
        f"  OOF F1 macro: {peg.get('oof_f1',       'N/A')}",
        f"  N classes   : {peg.get('n_classes',     'N/A')}",
        f"  Wall time   : {timing.get('peg', 0):.1f}s",
        "",
    ]

    # Temp
    tmp = metrics.get("temp", {})
    lines += [
        "── Temperature Prediction (regression) ─────────────────────",
        f"  OOF MAE  : {tmp.get('oof_mae',  'N/A')}",
        f"  OOF RMSE : {tmp.get('oof_rmse', 'N/A')}",
        f"  OOF R²   : {tmp.get('oof_r2',   'N/A')}",
        f"  Wall time: {timing.get('temp', 0):.1f}s",
        "",
    ]

    total = sum(timing.values())
    lines += [
        "=" * 64,
        f"Total training time: {total:.1f}s ({total/60:.1f} min)",
        "=" * 64,
    ]

    summary_path = RESULTS_DIR / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary → {summary_path}")
    print("\n".join(lines))


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save_plots(metrics: dict, timing: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # ── Plot 1: Metric comparison bar chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("CrystalNet — OOF Performance Summary", fontsize=14, fontweight="bold")

    # MAE for regression targets
    reg_targets = ["pH", "Salt conc.", "Temperature"]
    mae_vals = [
        metrics.get("ph", {}).get("oof_mae", 0) or 0,
        metrics.get("salt", {}).get("conc_oof_mae", 0) or 0,
        metrics.get("temp", {}).get("oof_mae", 0) or 0,
    ]
    colors_mae = ["#4C72B0", "#E8874A", "#845EC2"]
    axes[0].bar(reg_targets, mae_vals, color=colors_mae, edgecolor="white", linewidth=0.8)
    axes[0].set_title("OOF Mean Absolute Error")
    axes[0].set_ylabel("MAE")
    for i, v in enumerate(mae_vals):
        if v:
            axes[0].text(i, v + max(mae_vals) * 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(mae_vals) * 1.25 if any(mae_vals) else 1)
    axes[0].grid(axis="y", alpha=0.3)

    # R² for regression targets
    r2_vals = [
        metrics.get("ph", {}).get("oof_r2", 0) or 0,
        metrics.get("salt", {}).get("conc_oof_r2", 0) or 0,
        metrics.get("temp", {}).get("oof_r2", 0) or 0,
    ]
    axes[1].bar(reg_targets, r2_vals, color=colors_mae, edgecolor="white", linewidth=0.8)
    axes[1].set_title("OOF R² Score")
    axes[1].set_ylabel("R²")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for i, v in enumerate(r2_vals):
        if v:
            axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(axis="y", alpha=0.3)

    # Classification accuracy
    clf_targets = ["Salt type", "PEG type"]
    acc_vals = [
        metrics.get("salt", {}).get("type_accuracy", 0) or 0,
        metrics.get("peg",  {}).get("oof_accuracy",  0) or 0,
    ]
    colors_clf = ["#E8874A", "#27AE60"]
    axes[2].bar(clf_targets, acc_vals, color=colors_clf, edgecolor="white", linewidth=0.8)
    axes[2].set_title("Classification Accuracy")
    axes[2].set_ylabel("Accuracy")
    for i, v in enumerate(acc_vals):
        if v:
            axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {PLOTS_DIR / 'metrics_summary.png'}")

    # ── Plot 2: Training wall-time breakdown ──────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 4))
    t_labels = list(timing.keys())
    t_vals   = [timing[k] / 60 for k in t_labels]
    bar_colors = {"ph": "#4C72B0", "salt": "#E8874A", "peg": "#27AE60", "temp": "#845EC2"}
    bars = ax.barh(t_labels, t_vals, color=[bar_colors.get(k, "#999") for k in t_labels],
                   edgecolor="white")
    ax.set_xlabel("Minutes")
    ax.set_title("Training Wall Time per Pipeline")
    for bar, v in zip(bars, t_vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} min", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "training_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {PLOTS_DIR / 'training_time.png'}")

    # ── Plot 3: Meta-learner weight breakdown (pH example) ───────────────────
    ph_cfg = metrics.get("ph", {})
    if ph_cfg.get("meta_coef"):
        fig3, ax = plt.subplots(figsize=(5, 3))
        coefs  = ph_cfg["meta_coef"]
        labels = ["DL (ESM-2)", "XGBoost", "LightGBM"]
        colors = ["#4C72B0", "#E8874A", "#27AE60"]
        bars = ax.bar(labels, coefs, color=colors, edgecolor="white")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("pH Ridge Meta-Learner Weights")
        ax.set_ylabel("Weight")
        for bar, v in zip(bars, coefs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(coefs) * 0.03 if v >= 0 else v - max(coefs) * 0.06,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(PLOTS_DIR / "ph_meta_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved → {PLOTS_DIR / 'ph_meta_weights.png'}")


# ── Main orchestration ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CrystalNet Training Orchestrator")
    parser.add_argument("--epochs",  type=int, default=30,
                        help="Training epochs per pipeline (default: 30)")
    parser.add_argument("--folds",   type=int, default=5,
                        help="K-fold cross-validation folds (default: 5)")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                        choices=ALL_TARGETS, metavar="TARGET",
                        help="Which pipelines to train (default: all)")
    args = parser.parse_args()

    print("=" * 64)
    print("CrystalNet Training Orchestrator")
    print(f"Targets: {args.targets}")
    print(f"Epochs:  {args.epochs}   Folds: {args.folds}")
    print("=" * 64)

    timing = {}

    for target in args.targets:
        script = ROOT / target / "train.py"
        if not script.exists():
            print(f"\n[SKIP] {script} not found.")
            continue

        print(f"\n{'='*64}")
        print(f"TRAINING: {target.upper()}")
        print(f"{'='*64}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script),
             "--epochs", str(args.epochs),
             "--folds",  str(args.folds)],
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        timing[target] = elapsed

        if result.returncode != 0:
            print(f"[ERROR] {target} training exited with code {result.returncode}")
        else:
            print(f"\n[OK] {target} done in {elapsed:.1f}s")

    # Collect + report
    print("\n" + "=" * 64)
    print("Collecting metrics and generating plots…")
    metrics = _collect_metrics()

    # Save JSON
    output = {"metrics": metrics, "timing_seconds": timing,
              "trained_at": datetime.now().isoformat()}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics → {RESULTS_DIR / 'metrics.json'}")

    _write_summary(metrics, timing, args)
    _save_plots(metrics, timing)

    print("\nDone. Launch the app with:  python app.py")


if __name__ == "__main__":
    main()
