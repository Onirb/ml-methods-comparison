"""Empirical bias-variance decomposition on a known probabilistic problem.

The decomposition uses squared probability error (the Brier score), for which
the conditional class probability is the regression target.  Repeated fits use
independently generated training sets and a separate evaluation population.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GaussianBinaryProblem:
    """Balanced two-class Gaussian mixture with an analytic posterior."""

    means: FloatArray
    covariances: FloatArray
    class_prior: float = 0.5

    def __post_init__(self) -> None:
        if self.means.shape != (2, 2) or self.covariances.shape != (2, 2, 2):
            raise ValueError("This controlled problem requires two classes and two features.")
        if not 0.0 < self.class_prior < 1.0:
            raise ValueError("class_prior must be strictly between zero and one.")
        for covariance in self.covariances:
            if np.linalg.slogdet(covariance).sign <= 0:
                raise ValueError("Covariance matrices must be positive definite.")

    def sample(self, n_samples: int, seed: int) -> tuple[FloatArray, IntArray]:
        """Draw labelled observations without global random-state side effects."""

        if n_samples < 2:
            raise ValueError("n_samples must be at least two.")
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, self.class_prior, size=n_samples).astype(np.int64)
        X = np.empty((n_samples, 2), dtype=np.float64)
        for class_label in (0, 1):
            mask = y == class_label
            X[mask] = rng.multivariate_normal(
                self.means[class_label], self.covariances[class_label], mask.sum()
            )
        return X, y

    def posterior(self, X: FloatArray) -> FloatArray:
        """Return the exact P(Y=1 | X) implied by the Gaussian mixture."""

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n_samples, 2).")
        log_probabilities = []
        priors = (1.0 - self.class_prior, self.class_prior)
        for class_label, prior in enumerate(priors):
            covariance = self.covariances[class_label]
            difference = X - self.means[class_label]
            log_determinant = np.linalg.slogdet(covariance).logabsdet
            quadratic = np.einsum(
                "ni,ij,nj->n", difference, np.linalg.inv(covariance), difference
            )
            log_density = -0.5 * (2.0 * np.log(2.0 * np.pi) + log_determinant + quadratic)
            log_probabilities.append(np.log(prior) + log_density)
        log_0, log_1 = log_probabilities
        return 1.0 / (1.0 + np.exp(np.clip(log_0 - log_1, -700.0, 700.0)))


def shared_covariance_problem() -> GaussianBinaryProblem:
    """Return the same linear Gaussian geometry used by the modern baseline."""

    covariance = np.array([[1.0, 0.35], [0.35, 1.0]], dtype=np.float64)
    return GaussianBinaryProblem(
        means=np.array([[-1.25, -1.25], [1.25, 1.25]], dtype=np.float64),
        covariances=np.stack((covariance, covariance)),
    )


def estimate_bias_variance(
    models: Mapping[str, object],
    *,
    n_training_sets: int = 100,
    train_samples: int = 400,
    evaluation_samples: int = 10_000,
    seed: int = 20_261_119,
) -> pd.DataFrame:
    """Estimate the exact Brier-score decomposition for each fitted model.

    For a probabilistic prediction ``p_hat(X)`` and true posterior ``eta(X)``,
    expected Brier score decomposes as::

        E[(p_hat - Y)^2] = E[eta(1-eta)] + bias^2 + variance.

    The expectation over fitted models is approximated with independent
    training populations.  The evaluation population is generated separately.
    """

    if n_training_sets < 2:
        raise ValueError("At least two independent training sets are required.")
    if train_samples < 20 or evaluation_samples < 100:
        raise ValueError("Use at least 20 training and 100 evaluation observations.")
    problem = shared_covariance_problem()
    evaluation_X, _ = problem.sample(evaluation_samples, seed + 1)
    eta = problem.posterior(evaluation_X)
    irreducible_noise = float(np.mean(eta * (1.0 - eta)))

    rows: list[dict[str, float | int | str]] = []
    for model_name, template in models.items():
        fitted_probabilities = np.empty(
            (n_training_sets, evaluation_samples), dtype=np.float64
        )
        for training_index in range(n_training_sets):
            training_X, training_y = problem.sample(
                train_samples, seed + 10_000 + training_index
            )
            estimator = clone(template)
            estimator.fit(training_X, training_y)
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(f"Model {model_name!r} does not expose predict_proba().")
            fitted_probabilities[training_index] = estimator.predict_proba(evaluation_X)[:, 1]

        mean_prediction = fitted_probabilities.mean(axis=0)
        squared_bias = float(np.mean((mean_prediction - eta) ** 2))
        variance = float(np.mean((fitted_probabilities - mean_prediction) ** 2))
        expected_brier = irreducible_noise + squared_bias + variance
        direct_expected_brier = float(
            np.mean((fitted_probabilities - eta) ** 2) + irreducible_noise
        )
        rows.append({
            "problem": "shared_covariance_linear",
            "model": model_name,
            "training_sets": n_training_sets,
            "train_samples": train_samples,
            "evaluation_samples": evaluation_samples,
            "irreducible_noise": irreducible_noise,
            "squared_bias": squared_bias,
            "variance": variance,
            "expected_brier": expected_brier,
            "decomposition_residual": direct_expected_brier - expected_brier,
        })
    return pd.DataFrame(rows)
