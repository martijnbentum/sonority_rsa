# Partial RSA

## Requirements

- Compute partial RSA automatically when `compute_intensity_baseline=True`.
- Report `sonority_partial_rsa` for model–sonority RSA controlling for
  intensity and `intensity_partial_rsa` for model–intensity RSA controlling
  for sonority.
- Compute first-order partial Spearman correlation by average-ranking the
  three RDM upper triangles, residualizing model and target ranks separately
  against control ranks with an intercept, and correlating the residuals.
- Keep ordinary sonority and intensity RSA results.
- If either partial direction is undefined, record both as `NaN` and attach
  one shared reason identifying which direction or directions failed.
- Reuse the existing subset means and percentile confidence intervals; do
  not add p-values or permutation tests.
- Save, display, log, and plot both partial metrics. Plot partial series as
  dashed lines in the color of their corresponding ordinary RSA series.

## Tests

- Check the public partial-correlation function against residualized ranks,
  including tied distances and a confounded synthetic example.
- Cover incompatible, non-finite, and constant inputs.
- Verify paired invalidation, deterministic sampling, summaries, CSV output,
  diagnostics, display output, and run-log method metadata.
- Verify conditional plotting of both partial series in PNG and PDF output.

## Open Questions

None.
