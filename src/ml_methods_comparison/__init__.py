"""Reproducible synthetic experiments for classical classifiers."""

from .datasets import DATASET_BUILDERS, SyntheticDataset, make_dataset
from .evaluation import evaluate_models, summarize_metrics
from .models import build_models

__all__ = [
    "DATASET_BUILDERS",
    "SyntheticDataset",
    "build_models",
    "evaluate_models",
    "make_dataset",
    "summarize_metrics",
]
