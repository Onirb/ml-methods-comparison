import numpy as np

from ml_methods_comparison.datasets import make_dataset
from ml_methods_comparison.regularization import (
    evaluate_regularization,
    select_ccp_alpha,
    summarize_regularization,
)


def test_pruning_selection_uses_valid_candidates():
    train = make_dataset("xor_tree", n_samples=80, seed=1)
    validation = make_dataset("xor_tree", n_samples=80, seed=2)

    alpha, candidates = select_ccp_alpha(
        train.X, train.y, validation.X, validation.y, seed=3, max_candidates=6
    )

    assert len(candidates) <= 6
    assert alpha in candidates["ccp_alpha"].to_numpy()
    assert (candidates["validation_macro_f1"].between(0.0, 1.0)).all()
    assert (candidates["node_count"] >= 1).all()


def test_regularization_protocol_returns_one_untouched_test_result_per_strategy():
    metrics, selections = evaluate_regularization(
        dataset_names=["radial_rbf"],
        seeds=[0],
        train_samples=80,
        validation_samples=80,
        test_samples=120,
    )
    summary = summarize_regularization(metrics)

    assert len(metrics) == 5
    assert len(selections) == 1
    assert len(summary) == 5
    assert set(summary["macro_f1_count"]) == {1}
    assert np.isfinite(metrics[["accuracy", "balanced_accuracy", "macro_f1"]]).all().all()


def test_regularization_experiment_is_deterministic():
    arguments = dict(
        dataset_names=["shared_covariance_linear"],
        seeds=[2],
        train_samples=60,
        validation_samples=60,
        test_samples=100,
    )
    first_metrics, first_selections = evaluate_regularization(**arguments)
    second_metrics, second_selections = evaluate_regularization(**arguments)

    assert first_metrics.equals(second_metrics)
    assert first_selections.equals(second_selections)
