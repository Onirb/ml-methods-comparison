"""Evaluation helpers that share folds across every compared model."""

from collections.abc import Mapping

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .datasets import SyntheticDataset
from .models import build_models


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def shared_splits(
    dataset: SyntheticDataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> list[tuple[object, object]]:
    """Materialize one split plan so all models see identical folds."""

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    return list(splitter.split(dataset.X, dataset.y))


def evaluate_models(
    dataset: SyntheticDataset,
    models: Mapping[str, object] | None = None,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    cv_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate models on common repeated stratified folds."""

    selected_models = models if models is not None else build_models(cv_seed)
    splits = shared_splits(dataset, n_splits=n_splits, n_repeats=n_repeats, seed=cv_seed)
    rows: list[dict[str, object]] = []
    for split_index, (train_index, test_index) in enumerate(splits):
        repeat = split_index // n_splits
        fold = split_index % n_splits
        X_train, X_test = dataset.X[train_index], dataset.X[test_index]
        y_train, y_test = dataset.y[train_index], dataset.y[test_index]
        for model_name, template in selected_models.items():
            estimator = clone(template)
            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_test)
            rows.append({
                "dataset": dataset.name,
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro"),
            })
    return pd.DataFrame(rows)


def summarize_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate means and uncertainty across independent data seeds when available."""

    required = {"dataset", "model", *METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    aggregation_input = results
    if "data_seed" in results.columns:
        aggregation_input = (
            results.groupby(["dataset", "model", "data_seed"], as_index=False)[list(METRICS)]
            .mean()
        )
    summary = aggregation_input.groupby(["dataset", "model"])[list(METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()
