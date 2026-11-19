"""Controlled tree regularization and ensemble evaluation."""

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from .datasets import DATASET_BUILDERS, make_dataset


def select_ccp_alpha(
    X_train,
    y_train,
    X_validation,
    y_validation,
    *,
    seed: int,
    max_candidates: int = 25,
) -> tuple[float, pd.DataFrame]:
    """Select cost-complexity pruning strength on a dedicated validation set."""

    if max_candidates < 2:
        raise ValueError("max_candidates must be at least two.")
    path = DecisionTreeClassifier(random_state=seed).cost_complexity_pruning_path(
        X_train, y_train
    )
    # The last alpha collapses the tree to its root and is not a useful candidate.
    alphas = np.unique(path.ccp_alphas[:-1])
    if len(alphas) == 0:
        alphas = np.array([0.0])
    if len(alphas) > max_candidates:
        indices = np.linspace(0, len(alphas) - 1, max_candidates, dtype=int)
        alphas = np.unique(alphas[indices])

    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        model = DecisionTreeClassifier(random_state=seed, ccp_alpha=float(alpha))
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        rows.append({
            "ccp_alpha": float(alpha),
            "validation_macro_f1": f1_score(y_validation, prediction, average="macro"),
            "node_count": model.tree_.node_count,
            "depth": model.tree_.max_depth,
        })
    candidates = pd.DataFrame(rows)
    # Prefer the simpler tree when validation scores tie at saved precision.
    ranked = candidates.assign(
        rounded_score=candidates["validation_macro_f1"].round(12)
    ).sort_values(
        ["rounded_score", "node_count", "ccp_alpha"],
        ascending=[False, True, False],
    )
    return float(ranked.iloc[0]["ccp_alpha"]), candidates


def evaluate_regularization(
    *,
    dataset_names: Iterable[str] | None = None,
    seeds: Iterable[int] = range(5),
    train_samples: int = 600,
    validation_samples: int = 600,
    test_samples: int = 2_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed and selected tree strategies on untouched test populations."""

    names = list(dataset_names) if dataset_names is not None else list(DATASET_BUILDERS)
    seed_values = list(seeds)
    if not names or not seed_values:
        raise ValueError("At least one dataset and one seed are required.")
    unknown = set(names).difference(DATASET_BUILDERS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    metric_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    for data_seed in seed_values:
        for dataset_index, name in enumerate(names):
            base_seed = 100_000 * (data_seed + 1) + 1_000 * dataset_index
            train = make_dataset(name, train_samples, base_seed + 11)
            validation = make_dataset(name, validation_samples, base_seed + 22)
            test = make_dataset(name, test_samples, base_seed + 33)

            selected_alpha, candidates = select_ccp_alpha(
                train.X,
                train.y,
                validation.X,
                validation.y,
                seed=base_seed + 44,
            )
            selected = candidates.loc[
                np.isclose(candidates["ccp_alpha"], selected_alpha, rtol=0.0, atol=1e-15)
            ].iloc[0]
            selection_rows.append({
                "dataset": name,
                "data_seed": data_seed,
                "selected_ccp_alpha": selected_alpha,
                "validation_macro_f1": selected["validation_macro_f1"],
                "validation_node_count": int(selected["node_count"]),
                "validation_depth": int(selected["depth"]),
                "candidate_count": len(candidates),
            })

            combined_X = np.vstack((train.X, validation.X))
            combined_y = np.concatenate((train.y, validation.y))
            models = OrderedDict([
                ("decision_tree", DecisionTreeClassifier(random_state=base_seed + 55)),
                ("decision_tree_depth_3", DecisionTreeClassifier(
                    max_depth=3, random_state=base_seed + 55
                )),
                ("decision_tree_pruned", DecisionTreeClassifier(
                    ccp_alpha=selected_alpha, random_state=base_seed + 55
                )),
                ("random_forest", RandomForestClassifier(
                    n_estimators=200, random_state=base_seed + 55, n_jobs=1
                )),
                ("hist_gradient_boosting", HistGradientBoostingClassifier(
                    max_iter=100, max_leaf_nodes=15, random_state=base_seed + 55
                )),
            ])
            for model_name, model in models.items():
                model.fit(combined_X, combined_y)
                prediction = model.predict(test.X)
                row: dict[str, float | int | str] = {
                    "dataset": name,
                    "data_seed": data_seed,
                    "model": model_name,
                    "accuracy": accuracy_score(test.y, prediction),
                    "balanced_accuracy": balanced_accuracy_score(test.y, prediction),
                    "macro_f1": f1_score(test.y, prediction, average="macro"),
                }
                if isinstance(model, DecisionTreeClassifier):
                    row["node_count"] = model.tree_.node_count
                    row["depth"] = model.tree_.max_depth
                else:
                    row["node_count"] = np.nan
                    row["depth"] = np.nan
                metric_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows)


def summarize_regularization(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate test metrics across independent generated populations."""

    required = {"dataset", "model", "data_seed", "accuracy", "balanced_accuracy", "macro_f1"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    summary = metrics.groupby(["dataset", "model"])[
        ["accuracy", "balanced_accuracy", "macro_f1"]
    ].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()
