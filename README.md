# sonority-rsa

Bootstrap Representational Similarity Analysis (RSA) for wav2vec/HuBERT
phone middle frames sampled from syllables.

The pipeline connects three packages, with phraser and echoframe as the
canonical stores (nothing is duplicated into intermediate files):

1. [phraser](https://github.com/martijnbentum/phraser) provides the
   time-aligned syllable and phone annotations (an LMDB store);
2. [echoframe](https://github.com/martijnbentum/echoframe) provides the
   stored hidden-state embeddings, sliced to the middle frame of each
   phone;
3. [dutch_syllabifier](https://github.com/martijnbentum/dutch-syllabifier)
   provides the sonority weight of each phone label.

Each bootstrap sample selects syllables with replacement. All phone
middle frames belonging to each sampled syllable are included, so a
syllable sampled twice contributes its frames twice. For each layer, the
package builds:

1. a wav2vec/model RDM from hidden-state vectors using correlation distance;
2. a sonority RDM from phone sonority values using absolute distance.

The RSA score is the Spearman correlation between the upper triangles of
those RDMs. Repeating the bootstrap returns the mean RSA, percentile
confidence intervals, the raw per-bootstrap scores, and a run log that
makes every bootstrap draw replayable.

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

## Usage

Input is a list of phraser `Syllable` objects (with linked phones) and an
echoframe `Store`. Hidden states must already be stored in the echoframe
store at the phrase level; computing them is a separate step (see
`phraser.segment_embeddings`).

```python
from echoframe import Store
from phraser import Syllable, load_cache
from sonority_rsa import display_analysis, run_analysis, save_analysis

load_cache()
store = Store('cache')
syllables = list(Syllable.objects.filter(...))

summary, scores, log = run_analysis(
    syllables,
    model_name='wav2vec2',
    layers=[3, 7, 11],
    echoframe_store=store,
    n_syllables=100,
    n_bootstraps=1000,
    random_state=1,
)

display_analysis(summary, scores)
save_analysis(summary, scores, log, 'results/')
```

`run_analysis` fetches each layer's population once (one store read per
phrase, sliced to the middle frame of each phone) and then bootstraps in
memory. Syllables and phones that cannot be used are skipped and counted:
a syllable not linked to a phrase, a phrase without a stored embedding, a
phone label without a sonority class (e.g. silences), a phone no frames
overlap, and syllables left without usable phones.

The fetch and bootstrap steps are also available separately for
interactive work:

```python
from sonority_rsa import fetch_syllable_data, compute_bootstrap

population = fetch_syllable_data(syllables, 'wav2vec2', layer=7,
    echoframe_store=store)
scores = compute_bootstrap(population, n_syllables=100, n_bootstraps=1000,
    random_state=1)
```

A self-contained toy run (with fake stores, no LMDB needed) lives in
`examples/toy_analysis.py`.

## Outputs and Traceability

`save_analysis(summary, scores, log, out)` writes one directory per run:

- `summary.csv`: one row per layer with `run_id`, `layer`, `mean_rsa`,
  `ci_lower`, `ci_upper`, `n_bootstraps`, `n_syllables`
- `bootstrap_scores.csv`: one row per bootstrap with `run_id`, `layer`,
  `bootstrap`, `rsa`
- `run_log.json`: everything needed to trace and replay the run

The `run_id` appears in both CSV files and the log, so results stay
traceable when CSV files from several runs are combined.

The run log records the package version, all parameters (including the
master seed, which is drawn and logged when `random_state` is not given),
the echoframe store root, and per layer: the layer seed, the skip counts,
and the population syllable keys in fetched order (bytes keys are hex
encoded). Because each bootstrap consumes exactly one
`rng.integers(0, n_population, size=n_syllables)` draw, the sampled
syllables of every bootstrap can be recomputed from the log alone:

```python
from sonority_rsa import log_sampled_keys

draws = log_sampled_keys('results/run_log.json', layer=7)
draws[0]   # syllable keys drawn in bootstrap 0
```

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
