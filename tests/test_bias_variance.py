import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ml_methods_comparison.bias_variance import (
    estimate_bias_variance,
    shared_covariance_problem,
)


def test_known_posterior_is_valid_and_symmetric_at_origin():
    problem = shared_covariance_problem()
    probabilities = problem.posterior(np.array([[0.0, 0.0], [-2.0, -2.0], [2.0, 2.0]]))

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] == 0.5
    assert probabilities[1] < 0.5 < probabilities[2]


def test_independent_training_sets_close_brier_decomposition():
    models = {
        "logistic": LogisticRegression(max_iter=1_000),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
    }
    results = estimate_bias_variance(
        models,
        n_training_sets=4,
        train_samples=80,
        evaluation_samples=300,
        seed=123,
    )

    assert set(results["model"]) == set(models)
    assert (results[["irreducible_noise", "squared_bias", "variance"]] >= 0.0).all().all()
    assert np.allclose(results["decomposition_residual"], 0.0, atol=1e-12)
    assert (results["variance"] > 0.0).all()


def test_bias_variance_estimate_is_deterministic_for_fixed_seed():
    models = {"tree": DecisionTreeClassifier(max_depth=2, random_state=0)}
    arguments = dict(
        n_training_sets=3,
        train_samples=60,
        evaluation_samples=200,
        seed=456,
    )

    first = estimate_bias_variance(models, **arguments)
    second = estimate_bias_variance(models, **arguments)

    assert first.equals(second)
