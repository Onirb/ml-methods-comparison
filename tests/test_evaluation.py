import numpy as np
import pandas as pd
import pytest

from ml_methods_comparison.datasets import make_dataset
from ml_methods_comparison.evaluation import evaluate_models, shared_splits, summarize_metrics
from ml_methods_comparison.models import build_models


def test_shared_splits_are_stratified_and_cover_expected_repetitions():
    dataset = make_dataset("shared_covariance_linear", n_samples=120, seed=3)
    splits = shared_splits(dataset, n_splits=3, n_repeats=2, seed=11)
    assert len(splits) == 6
    for train, test in splits:
        assert not set(train).intersection(test)
        assert np.bincount(dataset.y[test]).tolist() == [20, 20]


def test_evaluation_is_deterministic_and_summarizes_all_models():
    dataset = make_dataset("radial_rbf", n_samples=240, seed=5)
    models = build_models(seed=13)
    first = evaluate_models(dataset, models, n_splits=3, n_repeats=1, cv_seed=17)
    second = evaluate_models(dataset, models, n_splits=3, n_repeats=1, cv_seed=17)
    assert len(first) == len(models) * 3
    assert first.equals(second)
    assert first[["accuracy", "balanced_accuracy", "macro_f1"]].notna().all().all()
    summary = summarize_metrics(first)
    assert set(summary["model"]) == set(models)


def test_radial_dataset_distinguishes_rbf_from_linear_boundary():
    dataset = make_dataset("radial_rbf", n_samples=400, seed=19)
    models = build_models(seed=23)
    selected = {name: models[name] for name in ("svm_linear", "svm_rbf")}
    results = evaluate_models(dataset, selected, n_splits=4, n_repeats=1, cv_seed=29)
    means = results.groupby("model")["macro_f1"].mean()
    assert means["svm_rbf"] > 0.95
    assert means["svm_rbf"] > means["svm_linear"] + 0.25


def test_summary_uncertainty_uses_independent_data_seeds_when_present():
    rows = []
    for data_seed, score in ((1, 0.6), (2, 0.8)):
        for fold in range(3):
            rows.append({
                "dataset": "controlled",
                "model": "model",
                "data_seed": data_seed,
                "fold": fold,
                "accuracy": score,
                "balanced_accuracy": score,
                "macro_f1": score,
            })
    summary = summarize_metrics(pd.DataFrame(rows)).iloc[0]
    assert summary["macro_f1_mean"] == pytest.approx(0.7)
    assert summary["macro_f1_count"] == 2
