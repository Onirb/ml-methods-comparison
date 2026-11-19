"""Run a correct empirical Brier-score bias-variance decomposition."""

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_methods_comparison.bias_variance import estimate_bias_variance
from ml_methods_comparison.models import build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-sets", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=400)
    parser.add_argument("--evaluation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_119)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modern"))
    return parser.parse_args()


def selected_models(seed: int) -> OrderedDict[str, object]:
    registry = build_models(seed)
    names = ("logistic_regression", "lda", "decision_tree_depth_3", "decision_tree")
    return OrderedDict((name, registry[name]) for name in names)


def plot_decomposition(results, output_path: Path) -> None:
    components = results.set_index("model")[[
        "irreducible_noise", "squared_bias", "variance"
    ]]
    labels = [name.replace("_", " ").title() for name in components.index]
    axis = components.plot.bar(
        stacked=True,
        figsize=(9, 5.5),
        color=["#9ca3af", "#f59e0b", "#2563eb"],
    )
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.set_ylabel("Expected Brier score component")
    axis.set_xlabel("")
    axis.set_title("Bias-variance decomposition on independent training sets")
    axis.legend(["Irreducible noise", "Squared bias", "Variance"], frameon=False)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)


def run(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = estimate_bias_variance(
        selected_models(args.seed),
        n_training_sets=args.training_sets,
        train_samples=args.train_samples,
        evaluation_samples=args.evaluation_samples,
        seed=args.seed,
    )
    results.to_csv(args.output_dir / "bias_variance.csv", index=False)
    plot_decomposition(results, args.output_dir / "bias_variance.png")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loss": "Brier score (squared probability error)",
        "problem": "balanced two-class Gaussian mixture with known posterior",
        "training_sets": args.training_sets,
        "train_samples": args.train_samples,
        "evaluation_samples": args.evaluation_samples,
        "seed": args.seed,
        "interpretation": "Descriptive controlled experiment; no model selection is performed.",
    }
    (args.output_dir / "bias_variance_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    args = parse_args()
    results = run(args)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
