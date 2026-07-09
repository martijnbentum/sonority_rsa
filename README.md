# sonority-rsa

Bootstrap Representational Similarity Analysis (RSA) for wav2vec/HuBERT
phone middle frames sampled from syllables.

The pipeline connects three packages:

1. [phraser](https://github.com/martijnbentum/phraser) provides the
   time-aligned phone and syllable annotations (an LMDB store);
2. [echoframe](https://github.com/martijnbentum/echoframe) provides the
   stored hidden-state embeddings, sliced to the middle frame of each
   phone;
3. [dutch_syllabifier](https://github.com/martijnbentum/dutch-syllabifier)
   provides the sonority weight of each phone label.

Each bootstrap sample selects syllable IDs with replacement. All
phone middle frames belonging to each sampled syllable are included, so a
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

## Extracting a Frame Table

`build_frame_table` walks phraser phrases and builds one row per phone,
holding the middle-frame hidden-state vector for each requested layer.
Hidden states must already be stored in the echoframe store (typically at
the phrase level; computing them is a separate step, see
`phraser.segment_embeddings`).

```python
from sonority_rsa import build_frame_table, save_frame_table

# phraser store with an echoframe store attached via
# echoframe_store.attach_phraser_store(source_id, phraser_store)
phrases = store.phrases

frames = build_frame_table(phrases, 'wav2vec2', layers=[3, 7, 11])
save_frame_table(frames, 'cache/frames.parquet')
```

Phones are skipped (and reported) when the phrase has no stored embedding
for a layer, no frames overlap the phone, the phone has no parent
syllable, or its label has no sonority class (e.g. silences).

The resulting frame table has one row per phone middle frame:

| column | description |
| --- | --- |
| `syllable_id` | phraser syllable key used for bootstrap sampling |
| `phone` | phone label |
| `sonority` | sonority weight from `dutch_syllabifier` (0-5) |
| `layer` | wav2vec/model layer |
| `vector` | middle-frame hidden-state vector |

`save_frame_table` / `load_frame_table` cache the table as Parquet, so
the slow LMDB extraction runs once per experiment.

## Python / IPython Usage

```python
from sonority_rsa import display_analysis, run_analysis, save_analysis

summary, scores = run_analysis(
    'cache/frames.parquet',   # or a frame table DataFrame
    n_syllables=100,
    n_bootstraps=1000,
    random_state=1,
)

display_analysis(summary, scores)
save_analysis(summary, scores, 'results/')
```

`run_analysis_from_stores` combines extraction and analysis in one call
and additionally returns the extracted frame table:

```python
from sonority_rsa import run_analysis_from_stores

summary, scores, frames = run_analysis_from_stores(
    phrases,
    'wav2vec2',
    layers=[3, 7, 11],
    n_syllables=100,
    n_bootstraps=1000,
    random_state=1,
)
```

The analysis helpers are intended for IPython and notebook workflows. They
return pandas DataFrames so you can inspect, filter, plot, or save results
from the same session. A self-contained toy run lives in
`examples/toy_analysis.py`.

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

Positive RSA values mean that pairs of phone middle frames that are close in
sonority tend to be close in wav2vec hidden-state geometry for that layer.
Values near zero mean little monotonic relationship between the two distance
structures in the sampled data.

The sonority scale is ordinal with six classes (stop, fricative, nasal,
liquid, glide, vowel), so the sonority RDM contains many tied distances.
Spearman correlation handles ties by midranking; expect the tie structure,
not a continuous predictor RDM.

Syllable position and sonority are structurally related. Position should not
simply be controlled away as a nuisance variable. Users may instead run
separate analyses by syllable template or position when that comparison is
scientifically appropriate.
