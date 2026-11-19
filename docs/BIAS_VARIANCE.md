# Corrected bias-variance experiment

## Why this experiment exists

The legacy notebook labelled repeated predictions from one already fitted tree
as a bias-variance simulation. Repeating the same prediction cannot measure
training-set variance: the fitted model never changes.

The corrected experiment answers a narrower and identifiable question:

> When training data change, how much of a model's expected probabilistic error
> comes from irreducible label uncertainty, systematic bias and fitted-model
> variance?

## Quantity being decomposed

For a probabilistic prediction `p_hat(X)` and the true class probability
`eta(X) = P(Y=1 | X)`, the expected Brier score has the exact decomposition:

```text
E[(p_hat(X) - Y)^2]
  = E[eta(X) * (1 - eta(X))]       irreducible noise
  + E[(E[p_hat(X)] - eta(X))^2]    squared bias
  + E[(p_hat(X) - E[p_hat(X)])^2]  variance
```

This is a probability-estimation decomposition, not a decomposition of
classification accuracy or zero-one error.

## Protocol

- Problem: the balanced shared-covariance Gaussian mixture used in the modern
  baseline.
- Ground truth: `eta(X)` is calculated analytically from the two Gaussian
  densities and equal class priors.
- Training variation: 100 independently generated labelled training sets,
  each with 400 observations.
- Evaluation: one separate population of 10,000 observations; it is never used
  for fitting or selection.
- Models: Logistic Regression, LDA, depth-3 Decision Tree and unrestricted
  Decision Tree.
- Selection: none. Configurations were fixed before the result was generated.

## Result

| Model | Noise | Bias squared | Variance | Expected Brier |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.048529 | 0.000347 | 0.000708 | 0.049585 |
| LDA | 0.048529 | 0.000025 | 0.000669 | 0.049223 |
| Decision Tree, depth 3 | 0.048529 | 0.001496 | 0.013877 | 0.063901 |
| Decision Tree, unrestricted | 0.048529 | 0.001001 | 0.050341 | 0.099871 |

The decomposition residual is zero to saved floating-point precision for every
model. LDA is well specified for this particular Gaussian geometry. The
unrestricted tree has much larger fitted-model variance, while limiting its
depth substantially reduces that component. This controlled result does not
establish a universal ranking outside the stated data-generating process.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.run_bias_variance
```

The command regenerates `bias_variance.csv`, `bias_variance_metadata.json` and
`bias_variance.png` under `results/modern/`.
