# Interpreting Partial and Multiple-Regression RSA

Both methods ask whether model geometry reflects sonority beyond its overlap
with intensity. They express the answer differently.

Partial RSA is currently implemented. Multiple-regression RSA is a proposed
complementary analysis.

| Method | Intuition | Main result | What it means |
| --- | --- | --- | --- |
| Partial RSA | Remove intensity-related structure from both the model and sonority RDMs, then correlate what remains. | Partial correlation | Strength of the remaining model–sonority rank association. |
| Multiple-regression RSA | Use sonority and intensity together to predict the model RDM. | Standardized β and incremental R² | Conditional slope for each predictor and the additional ranked-RDM variance it explains. |

## Partial RSA

`sonority_partial_rsa` is
`corr(model, sonority | intensity)`. `intensity_partial_rsa` reverses the
target and control.

- Positive: the model preserves the target geometry after controlling for the
  other predictor.
- Near zero: little target association remains after control.
- Negative: after control, model distances vary oppositely to target
  distances.
- Larger absolute values indicate stronger residual rank association. The
  value is bounded by -1 and 1.

The simplest intuition is **correlating the leftovers**. Partial RSA does not
measure a proportion of uniquely explained variance.

## Multiple-Regression RSA

A rank-based implementation consistent with this package would fit, per
subset and layer:

```text
rank(model RDM) = intercept
                + β_sonority * rank(sonority RDM)
                + β_intensity * rank(intensity RDM)
                + error
```

Interpret each output as follows:

- `sonority_beta` / `intensity_beta`: direction and conditional strength of
  that predictor while holding the other fixed. β is not unique variance.
- `model_r2`: ranked model-RDM variance explained jointly by both predictors.
- `sonority_delta_r2`: additional R² gained by adding sonority to an
  intensity-only model.
- `intensity_delta_r2`: additional R² gained by adding intensity to a
  sonority-only model.

Use incremental R², not β², to describe unique explained variance. Shared
variance belongs to the joint model and cannot be assigned uniquely to either
predictor.

The simplest intuition is **letting the predictors compete to explain the
same outcome**. β describes each predictor's fitted contribution; incremental
R² describes its added explanatory value.

## Relating the Methods

Partial RSA and multiple-regression RSA should usually agree on direction and
the broad layer pattern, but their magnitudes are not interchangeable. Partial
correlation normalizes by the residual variation left after control; β depends
on regression scaling; incremental R² measures improvement in model fit.

For a sonority–intensity RDM correlation around 0.34, the two-predictor
variance inflation factor is about 1.13, indicating little collinearity.
Calculate this per subset because predictor correlation changes with the
sample.

RDM cells are dependent, so ordinary OLS p-values and standard errors should
not be used. Interpret the package's repeated-subset distributions and
percentile intervals descriptively; formal inference would require an
appropriate nonparametric procedure.

## References

- [Nili et al. (2014), *A Toolbox for Representational Similarity Analysis*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3990488/)
- [Tarhan and Konkle (2020), multiple-regression RSA and collinearity checks](https://elifesciences.org/articles/47686)
