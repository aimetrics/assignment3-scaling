"""
Chinchilla IsoFLOPs scaling law analysis.

For each compute budget Ci, finds the model size Nopt(Ci) with lowest training loss,
then fits power laws Nopt ∝ C^a and Dopt ∝ C^b via log-log OLS and extrapolates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Run:
    parameters: float
    compute_budget: float
    final_loss: float


def load_runs(path: Path) -> list[Run]:
    return [Run(**item) for item in json.loads(path.read_text())]


def get_isoflops_optima(runs: Iterable[Run]) -> list[tuple[float, float, float]]:
    """Return (compute_budget, N_opt, D_opt) for each budget."""
    grouped: dict[float, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.compute_budget, []).append(run)

    optima: list[tuple[float, float, float]] = []
    for compute_budget, group in sorted(grouped.items()):
        best = min(group, key=lambda r: r.final_loss)
        d_opt = compute_budget / (6.0 * best.parameters)
        optima.append((compute_budget, best.parameters, d_opt))
    return optima


def fit_power_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Fit y = k * x^alpha via OLS in log-log space. Returns (k, alpha)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mean_x = sum(lx) / len(lx)
    mean_y = sum(ly) / len(ly)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(lx, ly, strict=True))
    var = sum((x - mean_x) ** 2 for x in lx)
    alpha = cov / var
    k = math.exp(mean_y - alpha * mean_x)
    return k, alpha


def predict_power_law(c: float, k: float, alpha: float) -> float:
    return k * c ** alpha


def plot_scaling_law(
    ax: plt.Axes,
    budgets: list[float],
    optima: list[float],
    k: float,
    alpha: float,
    predict_budgets: list[float],
    y_label: str,
    title: str,
    symbol: str,
) -> None:
    c_line = np.logspace(np.log10(min(budgets)), np.log10(max(predict_budgets)), 300)
    y_line = [predict_power_law(c, k, alpha) for c in c_line]

    ax.scatter(budgets, optima, color="steelblue", zorder=5,
               label=f"Data points $(C_i, {symbol}_{{opt}}(C_i))$")
    ax.plot(c_line, y_line, color="darkorange", linewidth=2,
            label=f"Fit: ${symbol}_{{opt}} \\propto C^{{{alpha:.3f}}}$")

    for target in predict_budgets:
        pred = predict_power_law(target, k, alpha)
        exp = int(round(math.log10(target)))
        ax.axvline(target, color="gray", linestyle="--", linewidth=0.8)
        ax.scatter([target], [pred], marker="*", s=120, zorder=6,
                   label=f"$C=10^{{{exp}}}$: ${symbol}_{{opt}}={pred:.2e}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget $C$ (FLOPs)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit IsoFLOPs scaling laws from synthetic run data.")
    parser.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--max-plot-budget", type=float, default=1e24)
    parser.add_argument("--predict-budgets", type=float, nargs="+", default=[1e23, 1e24])
    args = parser.parse_args()

    runs = load_runs(args.data)
    optima = get_isoflops_optima(runs)

    budgets = [x[0] for x in optima]
    n_opts = [x[1] for x in optima]
    d_opts = [x[2] for x in optima]

    n_k, n_alpha = fit_power_law(budgets, n_opts)
    d_k, d_alpha = fit_power_law(budgets, d_opts)

    print("IsoFLOPs optima (C, N_opt, D_opt):")
    for c, n, d in optima:
        print(f"  C={c:.3e}  N_opt={n:.4e}  D_opt={d:.4e}")

    print(f"\nN_opt power law: N = {n_k:.4e} * C^{n_alpha:.4f}")
    print(f"D_opt power law: D = {d_k:.4e} * C^{d_alpha:.4f}")

    print("\nPredictions:")
    predict_budgets = sorted(args.predict_budgets)
    for c in predict_budgets:
        print(f"  C={c:.0e}: N_opt={predict_power_law(c, n_k, n_alpha):.4e}  "
              f"D_opt={predict_power_law(c, d_k, d_alpha):.4e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_scaling_law(ax, budgets, n_opts, n_k, n_alpha, predict_budgets,
                     y_label="Optimal model size $N_{opt}$ (parameters)",
                     title="IsoFLOPs: Compute-Optimal Model Size Scaling Law",
                     symbol="N")
    plt.tight_layout()
    plt.savefig(args.output_dir / "isoflops_model_size_scaling.png", dpi=150)
    print(f"\nSaved {args.output_dir / 'isoflops_model_size_scaling.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_scaling_law(ax, budgets, d_opts, d_k, d_alpha, predict_budgets,
                     y_label="Optimal dataset size $D_{opt}$ (tokens)",
                     title="IsoFLOPs: Compute-Optimal Dataset Size Scaling Law",
                     symbol="D")
    plt.tight_layout()
    plt.savefig(args.output_dir / "isoflops_dataset_size_scaling.png", dpi=150)
    print(f"Saved {args.output_dir / 'isoflops_dataset_size_scaling.png'}")


if __name__ == "__main__":
    main()
