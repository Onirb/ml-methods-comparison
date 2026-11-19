"""Run the controlled classifier comparison and publish compact evidence."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from ml_methods_comparison.datasets import DATASET_BUILDERS, make_dataset
from ml_methods_comparison.evaluation import evaluate_models, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modern"))
    return parser.parse_args()


def plot_geometry(output_dir: Path, seed: int, samples: int) -> None:
    names = list(DATASET_BUILDERS)
    figure, axes = plt.subplots(1, len(names), figsize=(18, 3.5), constrained_layout=True)
    for axis, name in zip(axes, names, strict=True):
        dataset = make_dataset(name, n_samples=samples, seed=seed)
        axis.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y, cmap="coolwarm", s=8, alpha=0.7)
        axis.set_title(name.replace("_", "\n"), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Controlled synthetic datasets (seed={seed})")
    figure.savefig(output_dir / "dataset_geometry.png", dpi=160)
    plt.close(figure)


def plot_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    table = summary.pivot(index="model", columns="dataset", values="macro_f1_mean")
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt=".3f", vmin=0.45, vmax=1.0, cmap="viridis", ax=axis)
    axis.set_title("Macro F1 across controlled dataset geometries")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Model")
    figure.savefig(output_dir / "macro_f1_heatmap.png", dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not args.seeds:
        raise ValueError("At least one data seed is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for data_seed in args.seeds:
        for name in DATASET_BUILDERS:
            dataset = make_dataset(name, n_samples=args.samples, seed=data_seed)
            frame = evaluate_models(
                dataset,
                n_splits=args.splits,
                n_repeats=args.repeats,
                cv_seed=10_000 + data_seed,
            )
            frame.insert(1, "data_seed", data_seed)
            frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_geometry(args.output_dir, args.seeds[0], args.samples)
    plot_heatmap(summary, args.output_dir)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_seeds": args.seeds,
        "samples_per_dataset": args.samples,
        "cv_splits": args.splits,
        "cv_repeats": args.repeats,
        "datasets": list(DATASET_BUILDERS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "selection_rule": "No winner is selected; results describe behavior under fixed controlled conditions.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, summary


def main() -> None:
    args = parse_args()
    metrics, summary = run(args)
    print(f"Saved {len(metrics)} fold-level rows and {len(summary)} summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
