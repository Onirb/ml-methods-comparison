"""Run the held-out tree regularization and ensemble experiment."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ml_methods_comparison.datasets import DATASET_BUILDERS
from ml_methods_comparison.regularization import (
    evaluate_regularization,
    summarize_regularization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--validation-samples", type=int, default=600)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modern"))
    return parser.parse_args()


def plot_results(summary, output_path: Path) -> None:
    table = summary.pivot(index="model", columns="dataset", values="macro_f1_mean")
    table = table.rename(
        index={
            "decision_tree": "Decision Tree",
            "decision_tree_depth_3": "Decision Tree (depth 3)",
            "decision_tree_pruned": "Decision Tree (selected pruning)",
            "hist_gradient_boosting": "Histogram Gradient Boosting",
            "random_forest": "Random Forest",
        },
        columns={
            "heteroscedastic_qda": "Heteroscedastic",
            "indistinguishable_control": "Negative control",
            "radial_rbf": "Radial rings",
            "shared_covariance_linear": "Linear Gaussian",
            "xor_tree": "XOR clusters",
        },
    )
    figure, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt=".3f", vmin=0.45, vmax=1.0, cmap="mako", ax=axis)
    axis.set_title("Held-out macro F1: tree regularization and ensembles")
    axis.set_xlabel("Untouched test population")
    axis.set_ylabel("Model strategy")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, selections = evaluate_regularization(
        seeds=args.seeds,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
    )
    summary = summarize_regularization(metrics)
    metrics.to_csv(args.output_dir / "regularization_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "regularization_summary.csv", index=False)
    selections.to_csv(args.output_dir / "pruning_selections.csv", index=False)
    plot_results(summary, args.output_dir / "regularization_heatmap.png")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_seeds": args.seeds,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "test_samples": args.test_samples,
        "datasets": list(DATASET_BUILDERS),
        "selection": "ccp_alpha selected by macro F1 on a dedicated validation population",
        "final_fit": "models fitted on train plus validation after selection",
        "test_boundary": "test populations are used once for final evaluation",
    }
    (args.output_dir / "regularization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, summary, selections


def main() -> None:
    args = parse_args()
    metrics, summary, selections = run(args)
    print(
        f"Saved {len(metrics)} held-out evaluations, {len(summary)} summaries "
        f"and {len(selections)} pruning selections to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
