# Methodology audit of the original notebook

The original assignment is useful academic evidence, but several outputs cannot
support the strength of the conclusions originally attached to them. This audit
preserves that work while defining the boundary between historical results and
the modern experiment.

## Critical findings

1. **The original bias–variance block does not estimate estimator variance.**
   A decision tree is fitted once and its prediction on the same training data
   is repeated 100 times. Those predictions are identical, so the measured
   variance is structurally zero. A valid experiment must fit new models on
   independent training samples and predict a fixed independent evaluation set.
2. **The same test split is used to compare pruning values.** `ccp_alpha` is
   effectively selected on the test set. Model selection must happen inside the
   training data, followed by one independent test evaluation.
3. **Some later cells depend on silently overwritten notebook variables.** The
   meaning of `X_circular` and `X_poly` depends on which earlier loop ran last.
4. **Cross-validation does not shuffle ordered synthetic observations.** Folds
   may represent contiguous angle or x-coordinate regions rather than IID
   samples. This could be a deliberate extrapolation test, but it was not
   defined as one.

## Important limitations

- The original “uniform” dataset gives both labels the same distribution. Its
  chance-level score is an appropriate negative control, not model failure.
- Global calls to `np.random.seed(42)` couple experiments and do not estimate
  uncertainty across independent generated datasets.
- Accuracy is the dominant metric. Balanced accuracy, macro F1 and dispersion
  across seeds are required for the modern comparison.
- Scale-sensitive models were compared without a fold-local scaler.
- Saved heatmap values were transcribed manually rather than generated from a
  result table.
- Broad statements about the “best” model exceed a single realization with
  default hyperparameters.

## Modern protocol

The replacement experiment uses five explicit roles:

- shared-covariance Gaussian classes for linear models and LDA;
- class-specific covariance geometry for QDA;
- XOR clusters for trees and nonlinear models;
- concentric rings for RBF SVM;
- identical class distributions as a negative control.

Datasets use local `numpy.random.Generator` instances. Every model sees the same
`RepeatedStratifiedKFold` splits. Scaling is contained inside pipelines, so it is
fitted only on each training fold. Fold-level accuracy, balanced accuracy and
macro F1 are saved before aggregation. Results are observations under fixed
conditions, not a universal ranking of algorithms.
