"""Controlled two-dimensional datasets with explicit inductive-bias goals."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated binary classification problem and its intended role."""

    name: str
    X: FloatArray
    y: IntArray
    hypothesis: str

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != 2:
            raise ValueError("Synthetic datasets must have exactly two features.")
        if self.y.ndim != 1 or len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of samples.")
        if set(np.unique(self.y)) != {0, 1}:
            raise ValueError("Synthetic datasets must contain binary labels 0 and 1.")


def _validate_size(n_samples: int, divisor: int = 2) -> None:
    if n_samples < 40 or n_samples % divisor:
        raise ValueError(f"n_samples must be at least 40 and divisible by {divisor}.")


def _shuffle(X: FloatArray, y: IntArray, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
    order = rng.permutation(len(y))
    return X[order], y[order]


def shared_covariance_linear(n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Gaussian classes with shared covariance and a linear boundary."""

    _validate_size(n_samples)
    rng = np.random.default_rng(seed)
    per_class = n_samples // 2
    covariance = np.array([[1.0, 0.35], [0.35, 1.0]])
    class_0 = rng.multivariate_normal([-1.25, -1.25], covariance, per_class)
    class_1 = rng.multivariate_normal([1.25, 1.25], covariance, per_class)
    X = np.vstack((class_0, class_1)).astype(np.float64)
    y = np.repeat([0, 1], per_class).astype(np.int64)
    X, y = _shuffle(X, y, rng)
    return SyntheticDataset(
        "shared_covariance_linear", X, y,
        "Linear and shared-covariance classifiers should be well matched.",
    )


def heteroscedastic_qda(n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Centered Gaussian classes separated mainly by covariance geometry."""

    _validate_size(n_samples)
    rng = np.random.default_rng(seed)
    per_class = n_samples // 2
    class_0 = rng.multivariate_normal([0.0, 0.0], [[0.20, 0.0], [0.0, 2.0]], per_class)
    class_1 = rng.multivariate_normal([0.0, 0.0], [[2.0, 0.0], [0.0, 0.20]], per_class)
    X = np.vstack((class_0, class_1)).astype(np.float64)
    y = np.repeat([0, 1], per_class).astype(np.int64)
    X, y = _shuffle(X, y, rng)
    return SyntheticDataset(
        "heteroscedastic_qda", X, y,
        "QDA can use class-specific covariance where a linear boundary cannot.",
    )


def xor_tree(n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Four compact clusters arranged as an XOR decision problem."""

    _validate_size(n_samples, divisor=4)
    rng = np.random.default_rng(seed)
    per_cluster = n_samples // 4
    centers = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    labels = np.array([0, 0, 1, 1])
    chunks = [rng.normal(center, 0.35, size=(per_cluster, 2)) for center in centers]
    X = np.vstack(chunks).astype(np.float64)
    y = np.repeat(labels, per_cluster).astype(np.int64)
    X, y = _shuffle(X, y, rng)
    return SyntheticDataset(
        "xor_tree", X, y,
        "Trees, nonlinear kernels and quadratic covariance models can capture XOR structure.",
    )


def radial_rbf(n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Concentric noisy rings with a radial nonlinear boundary."""

    _validate_size(n_samples)
    rng = np.random.default_rng(seed)
    per_class = n_samples // 2
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    inner_radius = rng.normal(0.85, 0.12, per_class)
    outer_radius = rng.normal(1.75, 0.16, per_class)
    radii = np.concatenate((inner_radius, outer_radius))
    X = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).astype(np.float64)
    y = np.repeat([0, 1], per_class).astype(np.int64)
    X, y = _shuffle(X, y, rng)
    return SyntheticDataset(
        "radial_rbf", X, y,
        "RBF and quadratic boundaries should represent the rings better than a linear model.",
    )


def indistinguishable_control(n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Negative control: both labels are sampled from the same distribution."""

    _validate_size(n_samples)
    rng = np.random.default_rng(seed)
    per_class = n_samples // 2
    class_0 = rng.normal(0.0, 1.0, size=(per_class, 2))
    class_1 = rng.normal(0.0, 1.0, size=(per_class, 2))
    X = np.vstack((class_0, class_1)).astype(np.float64)
    y = np.repeat([0, 1], per_class).astype(np.int64)
    X, y = _shuffle(X, y, rng)
    return SyntheticDataset(
        "indistinguishable_control", X, y,
        "No classifier should generalize materially above chance.",
    )


DATASET_BUILDERS: dict[str, Callable[[int, int], SyntheticDataset]] = {
    "shared_covariance_linear": shared_covariance_linear,
    "heteroscedastic_qda": heteroscedastic_qda,
    "xor_tree": xor_tree,
    "radial_rbf": radial_rbf,
    "indistinguishable_control": indistinguishable_control,
}


def make_dataset(name: str, n_samples: int = 600, seed: int = 0) -> SyntheticDataset:
    """Generate a named dataset with no global random-state side effects."""

    try:
        builder = DATASET_BUILDERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {choices}.") from exc
    return builder(n_samples, seed)
