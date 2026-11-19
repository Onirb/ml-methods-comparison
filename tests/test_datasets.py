import numpy as np
import pytest

from ml_methods_comparison.datasets import DATASET_BUILDERS, make_dataset


@pytest.mark.parametrize("name", sorted(DATASET_BUILDERS))
def test_datasets_are_balanced_reproducible_and_two_dimensional(name):
    first = make_dataset(name, n_samples=200, seed=7)
    second = make_dataset(name, n_samples=200, seed=7)
    assert first.X.shape == (200, 2)
    assert first.y.shape == (200,)
    assert np.bincount(first.y).tolist() == [100, 100]
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)


def test_seed_changes_generated_values_without_changing_contract():
    first = make_dataset("radial_rbf", n_samples=200, seed=1)
    second = make_dataset("radial_rbf", n_samples=200, seed=2)
    assert not np.array_equal(first.X, second.X)
    assert np.bincount(second.y).tolist() == [100, 100]


def test_xor_requires_complete_clusters():
    with pytest.raises(ValueError, match="divisible by 4"):
        make_dataset("xor_tree", n_samples=202, seed=0)


def test_unknown_dataset_has_actionable_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        make_dataset("not-a-dataset")
