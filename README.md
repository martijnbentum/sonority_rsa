# sonority-rsa

Bootstrap Representational Similarity Analysis (RSA) for wav2vec
phone-center frames sampled from syllables.

Each bootstrap sample selects syllable IDs with replacement. All
phone-center frames belonging to each sampled syllable are included, so a
syllable sampled twice contributes its frames twice. For each layer, the
package builds:

1. a wav2vec/model RDM from hidden-state vectors using correlation distance;
2. a sonority RDM from phone sonority values using absolute distance.

The RSA score is the Spearman correlation between the upper triangles of
those RDMs. Repeating the bootstrap returns the mean RSA, percentile
confidence intervals, and the raw per-bootstrap scores.

This analysis tests whether wav2vec frame geometry reflects sonority across
sampled syllable sets.

## Install

Create and activate a uv virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install this package:

```bash
uv pip install -e .
```

The project depends on `phraser`, `echoframe`, and `dutch-syllabifier`, using
the git install forms documented by those repositories:

```bash
uv pip install git+https://github.com/martijnbentum/phraser.git
uv pip install git+https://github.com/martijnbentum/echoframe.git
uv pip install git+https://github.com/martijnbentum/dutch-syllabifier.git
```

## Input Format

Input can be CSV or Parquet with one row per phone-center frame. Required
columns:

| column | description |
| --- | --- |
| `syllable_id` | syllable identifier used for bootstrap sampling |
| `phone` | phone label |
| `sonority` | numeric sonority value |
| `layer` | wav2vec/model layer |
| `vector` | hidden-state vector |

The `vector` column may contain JSON-like lists such as `[0.1, 0.2]` or
space-separated floats such as `0.1 0.2`.

The table is normally prepared from `phraser` annotations, hidden-state vectors
stored with `echoframe`, and sonority values computed with
`dutch_syllabifier`.

## Python / IPython Usage

```python
from sonority_rsa.analysis import (display_analysis, run_analysis,
    save_analysis)

summary, scores = run_analysis(
    'examples/toy_frames.csv',
    n_syllables=3,
    n_bootstraps=100,
    random_state=1,
)

display_analysis(summary, scores)
save_analysis(summary, scores, 'results/')
```

The analysis helpers are intended for IPython and notebook workflows. They
return pandas DataFrames so you can inspect, filter, plot, or save results
from the same session.

Saved outputs:

- `results/summary.csv`
- `results/bootstrap_scores.csv`

`summary.csv` contains one row per layer:

- `layer`
- `mean_rsa`
- `ci_lower`
- `ci_upper`
- `n_bootstraps`
- `n_syllables`

`bootstrap_scores.csv` contains:

- `layer`
- `bootstrap`
- `rsa`

## Interpretation

Positive RSA values mean that pairs of phone-center frames that are close in
sonority tend to be close in wav2vec hidden-state geometry for that layer.
Values near zero mean little monotonic relationship between the two distance
structures in the sampled data.

Syllable position and sonority are structurally related. Position should not
simply be controlled away as a nuisance variable. Users may instead run
separate analyses by syllable template or position when that comparison is
scientifically appropriate.
