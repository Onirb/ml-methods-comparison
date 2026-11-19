# Tree regularization and ensemble experiment

## Question

Can explicit tree regularization or aggregation improve held-out performance
without selecting a model on the final test data?

## Protocol

For each of five independent data seeds and each controlled geometry:

1. generate 600 training observations;
2. generate a separate 600-observation validation population;
3. select `ccp_alpha` from at most 25 candidates by validation macro F1;
4. refit the selected pruned tree on training plus validation data;
5. evaluate every strategy once on an untouched 2,000-observation test
   population.

The unrestricted tree, fixed depth-3 tree, Random Forest and Histogram
Gradient Boosting configurations are fixed. The test population is not used to
select `ccp_alpha`, models or thresholds.

## Mean held-out macro F1

| Dataset | Full tree | Depth 3 | Selected pruning | Hist. boosting | Random Forest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared covariance linear | 0.898 | 0.921 | 0.921 | 0.922 | 0.925 |
| Heteroscedastic covariance | 0.751 | 0.741 | 0.787 | 0.784 | 0.776 |
| XOR clusters | 0.991 | 0.848 | 0.992 | 0.992 | 0.995 |
| Radial rings | 0.992 | 0.890 | 0.992 | 0.994 | 0.994 |
| Indistinguishable control | 0.497 | 0.452 | 0.419 | 0.494 | 0.506 |

Values are means across five independently generated test populations. Full
means, seed-level standard deviations and selections are saved as CSV.

## Interpretation

- Pruning substantially improves the tree on the heteroscedastic problem in
  this protocol, while Random Forest is strongest by a small margin on the
  linear, XOR and radial geometries.
- A fixed maximum depth of three is not a generally safe shortcut. It works
  reasonably on the linear geometry but underfits the radial and XOR problems;
  its XOR result is also unstable across seeds.
- The negative control remains the most important guardrail. Random Forest,
  boosting and the full tree remain near chance on the independent test data.
- Validation-based pruning is unstable when there is no real signal. In two of
  five negative-control seeds, refitting with the selected alpha produced a
  root-only classifier and macro F1 of 0.333. This is a warning about selection
  noise, not evidence that the data are predictable.
- These results describe fixed configurations on synthetic geometries. They do
  not establish a universal ensemble ranking.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.run_regularization
```

The command regenerates the detailed test metrics, summary, pruning selections,
metadata and heatmap under `results/modern/`.
