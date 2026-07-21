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

Each sample draws a subset of distinct syllables without replacement, so
every syllable in one sample is unique and no exact-duplicate rows enter
the RDMs (across samples the same syllable can reappear). `subset_size`
must be at least 30 and cannot exceed the layer population. Repeated
full-population draws are rejected: when `subset_size` equals the
population size, use exactly one subset. Subsets are sampled independently,
not as disjoint partitions. A warning is emitted when `subset_size` is at
least 50% of the population, because repeated draws may overlap heavily and
have limited diversity. All phone middle frames belonging to each sampled
syllable are included. For each layer, the package builds:

1. a wav2vec/model RDM from hidden-state vectors using correlation distance;
2. a sonority RDM from phone sonority values using absolute distance.

Optionally, the package also builds an intensity RDM from centered phone
intensities and reports its RSA plus direct and RDM-level Spearman
correlations between sonority and intensity. The intensity-enabled analysis
also reports partial RSA in both directions: model–sonority controlling for
intensity and model–intensity controlling for sonority.

The RSA score is the Spearman correlation between the upper triangles of
those RDMs. Repeating over many samples returns the mean RSA, percentile
confidence intervals, the raw per-subset scores, and a run log that makes
every sampled subset replayable.

The `ci` argument is a confidence level expressed as a fraction; its default
of `0.95` produces a 95% percentile interval. Values below `0.90` are
accepted with a warning, while values outside `(0, 1)` are rejected.

This analysis tests whether wav2vec frame geometry reflects sonority across
sampled syllable sets.

## Install

### Install with `uv pip`

```bash
uv pip install git+https://github.com/martijnbentum/sonority_rsa.git
```

### Install with `pip`

```bash
pip install git+https://github.com/martijnbentum/sonority_rsa.git
```

The Git-based runtime dependencies `phraser`, `echoframe`, and
`dutch-syllabifier` are declared in [`pyproject.toml`](./pyproject.toml)
and installed automatically. Installation requires Git and network access
to GitHub.

### Editable install

```bash
git clone git@github.com:martijnbentum/sonority_rsa.git
cd sonority_rsa
uv venv
uv sync
```

If the uv virtual environment does not exist yet, `uv venv` creates
`.venv` and `uv sync` installs the package with all dependencies
(including the `dev` group with pytest).

## Development

Activate the version-bump pre-commit hook once per clone:

```bash
git config core.hooksPath .githooks
```

The hook bumps the patch version in `pyproject.toml` on every commit and
keeps `uv.lock` out of the working tree.

Run the tests:

```bash
.venv/bin/python -m pytest
```

## Usage

Input is a list of phraser `Syllable` objects (with linked phones) and an
echoframe `Store`. Hidden states must already be stored in the echoframe
store at the phrase level; computing them is a separate step (see
`phraser.segment_embeddings`).

```python
from echoframe import Store
from phraser import Syllable, load_cache
from sonority_rsa import (display_analysis, plot_analyses, plot_analysis,
    run_analysis, save_analysis)

load_cache()
store = Store('cache')
syllables = list(Syllable.objects.filter(...))

summary, results, log = run_analysis(
    syllables,
    model_name='wav2vec2',
    layers=[3, 7, 11],
    echoframe_store=store,
    subset_size=100,
    n_subsets=1000,
    random_state=1,
    compute_random_baseline=True,
    compute_intensity_baseline=True,
)

results['rsa'][7]
results['random_baseline_rsa'][7]
results['intensity_rsa'][7]
results['sonority_intensity_correlation'][7]
results['sonority_intensity_rdm_correlation'][7]
results['sonority_partial_rsa'][7]
results['intensity_partial_rsa'][7]
display_analysis(summary, results)
save_analysis(summary, results, log, 'results/')
plot_analysis('results/', title='RSA by model layer')
plot_analysis('results/', title='RSA by model layer', filetype='.pdf')
plot_analyses(
    ['results/model_a/', 'results/model_b/'],
    ['Model A', 'Model B'],
    filetype='.pdf',
)
```

`run_analysis` fetches each layer's population once (one store read per
phrase, sliced to the middle frame of each phone) and then samples in
memory. Syllables and phones that cannot be used are skipped and counted:
a syllable not linked to a phrase, a phrase without a stored embedding, a
phone label without a sonority class (e.g. silences), a phone no frames
overlap, a phone whose stored vector is constant across features (a
broken/zeroed embedding), and syllables left without usable phones. A
layer whose usable phones all share one sonority class has no variation to
correlate against, so it is dropped like any other unusable layer (issue
recorded under `failed_layers`).

The analysis assumes that usable syllables are the same for every requested
layer. Consequently, an invalid `subset_size` is a configuration error for
the full run, rather than a reason to drop an individual layer.

When `compute_random_baseline=True`, every sampled subset produces two paired
scores: its observed RSA and an RSA after shuffling that subset's sonority
values once. Both scores reuse the same model RDM. The shuffle uses a separate
generator whose seed is deterministically derived from the layer seed, so it
does not change syllable sampling or logged replay. Results are grouped first
by metric and then by layer; without the flag, `results` contains only the
`rsa` key.

When `compute_intensity_baseline=True`, the complete audio interval of every
phone with a usable vector is converted to one RMS-power intensity value:
`10 * log10(mean(signal ** 2) / 4e-10)`. Values are centered per recording by
subtracting the mean intensity of all usable phones from that recording.
This preserves within-recording distances while removing recording-level
gain offsets. Raw intensities are cached across layers. Each subset adds
`intensity_rsa`, a direct phone-value Spearman correlation between sonority
and intensity, and a Spearman correlation between their RDM upper triangles.
It also adds `sonority_partial_rsa`, controlling the model–sonority
association for intensity distances, and `intensity_partial_rsa`, controlling
the model–intensity association for sonority distances. All predictors reuse
the same model RDM.

Partial RSA is computed over the three RDM upper triangles. Each triangle is
average-ranked, the model and target ranks are separately residualized against
the control ranks with an intercept, and the two residual vectors are
correlated. If either direction is undefined, both partial scores for that
subset are recorded as `NaN` and excluded from their summaries.

Intensity is evaluated only after a phone passes the embedding checks. A
missing or unreadable audio file, invalid interval, empty segment, zero-power
segment, or non-finite value for such a phone aborts the run. The error lists
the audio filename, phone label, and start/end times. A phone already skipped
for a missing or invalid vector does not require audio.

Individual sampled subsets that happen to contain one sonority class have
no sonority-distance variation. Their RSA score is recorded as `NaN` and is
excluded from the summary; `n_subsets_valid` reports the resulting effective
sample size. A layer whose subsets are all invalid is dropped and recorded
under `failed_layers` with its invalid-subset diagnostics.

The fetch and sampling steps are also available separately for
interactive work. A single random baseline score can be computed by
shuffling the observed sonority values while leaving the vectors unchanged;
the shuffle preserves the sonority class counts and tied-distance structure.

```python
import numpy as np

from sonority_rsa import (compute_rsa_scores, compute_sonority_random_baseline,
    compute_sonority_rsa, fetch_syllable_data, sample_syllables)

population = fetch_syllable_data(syllables, 'wav2vec2', layer=7,
    echoframe_store=store)
scores = compute_rsa_scores(population, subset_size=100, n_subsets=1000,
    random_state=1)

rng = np.random.default_rng(1)
vectors, sonority, _ = sample_syllables(population, subset_size=100, rng=rng)
observed = compute_sonority_rsa(vectors, sonority)
baseline = compute_sonority_random_baseline(vectors, sonority,
    random_state=42)
```

The residualized-rank calculation is also available directly as
`partial_spearman_rsa(model_rdm, target_rdm, control_rdm)`.

Phone intensity functions are available separately. `compute_phone_intensity`
implements the RMS-power measure used by the intensity baseline.
`compute_praat_intensity` computes a genuine Praat intensity contour over the
complete recording and returns its energy-domain average for the phone, so
the two raw measures can be compared directly.

```python
from sonority_rsa import compute_phone_intensity, compute_praat_intensity

rms_db = compute_phone_intensity(phone)
praat_db = compute_praat_intensity(phone, minimum_pitch=100)
```

The Praat implementation uses
[`praat-parselmouth`](https://parselmouth.readthedocs.io/en/stable/api_reference.html)
with mean-pressure subtraction enabled and Praat's default contour time step.

A self-contained toy run (with fake stores, no LMDB needed) lives in
`examples/toy_analysis.py`.

## Outputs and Traceability

`save_analysis(summary, results, log, out)` writes one directory per run:

- `summary.csv`: one row per layer with `run_id`, `layer`, `mean_rsa`,
  `ci_lower`, `ci_upper`, `n_subsets`, `n_subsets_valid`, `subset_size`
  (`n_subsets` is the number of subsets drawn; `n_subsets_valid` is how
  many were non-NaN, i.e. the effective sample behind `mean_rsa` and the
  interval). With the random baseline enabled, it also contains the baseline
  mean and interval and the mean paired RSA difference and its interval. With
  the intensity baseline enabled, it contains means and intervals for
  `intensity_rsa`, `sonority_intensity_correlation`, and
  `sonority_intensity_rdm_correlation`, plus `sonority_partial_rsa` and
  `intensity_partial_rsa`. `n_subsets_partial_valid` gives the shared valid
  subset count for the paired partial results.
- `rsa_scores.csv`: one row per subset with `run_id`, `layer`, `subset`,
  `rsa`, and `invalid_reason` (blank for valid scores). With the random
  baseline enabled, `random_baseline_rsa` and the paired `rsa_difference`
  are included. With the intensity baseline enabled, the three intensity
  metrics, both partial RSA metrics, `intensity_invalid_reason`, and
  `partial_invalid_reason` are included; no RSA-minus-intensity difference is
  reported.
- `run_log.json`: everything needed to trace and replay the run
- `rsa_by_layer.png` or `rsa_by_layer.pdf`: created by
  `plot_analysis(output_dir, title, filetype='.png', show_plot=True)` from the
  two CSV files.
  It shows subset-level RSA values together with per-layer means and
  confidence intervals for sonority, plus intensity, partial RSA, and random
  baseline series when those optional results are available.
- `rsa_by_layer_panels.png` or `rsa_by_layer_panels.pdf`: created by
  `plot_analyses(output_dirs, titles, filetype='.png', show_plot=True)` in the
  parent of the first analysis directory. It shows one analysis per panel and
  uses the requested file type for the combined figure.

Both plotting functions turn on Matplotlib interactive mode and show the
figure by default. Pass `show_plot=False` to save without displaying it.

The `run_id` appears in both CSV files and the log, so results stay
traceable when CSV files from several runs are combined.

The run log records the package version, all parameters (including the
master seed, which defaults to 42 and can be overridden with
`random_state`), whether either optional baseline was enabled, the echoframe
store root, and per layer: the layer seed, the derived random-baseline seed
when applicable, the skip counts,
and the population syllable keys in fetched order (bytes keys are hex
encoded), plus invalid-subset counts and a per-subset reason list. A layer
with no usable data is dropped from `summary` and `results` and recorded
instead under the log's `failed_layers` key (its seed and the reason); every
requested layer still consumes one seed draw, so a dropped layer leaves the
surviving layers' seeds unchanged. A run in which every layer fails raises
`ValueError`. Because each subset consumes exactly one
`rng.choice(n_population, size=subset_size, replace=False)` draw, the sampled
syllables of every subset can be recomputed from the log alone:

```python
from sonority_rsa import log_sampled_keys

draws = log_sampled_keys('results/run_log.json', layer=7)
draws[0]   # syllable keys drawn in subset 0
```

## Interpretation

Positive RSA values mean that pairs of phone middle frames that are close in
sonority tend to be close in wav2vec hidden-state geometry for that layer.
Values near zero mean little monotonic relationship between the two distance
structures in the sampled data.

`intensity_rsa` measures whether pairs of phones with similar centered
intensity also have similar hidden-state geometry. The direct
sonority/intensity correlation describes whether more sonorous phones tend to
be more or less intense. The RDM correlation describes whether phone pairs
that differ in sonority also tend to differ in intensity; it is the more
direct indicator of overlap between the two RSA predictors. These descriptive
comparisons do not by themselves estimate unique association or control
intensity in the sonority RSA.

`sonority_partial_rsa` asks how much model–sonority rank association remains
after controlling for intensity distances. `intensity_partial_rsa` asks the
reverse question after controlling for sonority distances. These are
descriptive unique rank associations, not proportions of variance explained.
The package summarizes their repeated-subset distributions and does not report
ordinary correlation p-values because pairwise RDM cells are dependent.

See the [partial and multiple-regression RSA interpretation
guide](docs/INTERPRETING_PARTIAL_AND_MR_RSA.md) for a concise comparison of
partial correlations, regression coefficients, and incremental explained
variance.

The sonority scale is ordinal with six classes (stop, fricative, nasal,
liquid, glide, vowel), so the sonority RDM contains many tied distances.
Spearman correlation handles ties by midranking; expect the tie structure,
not a continuous predictor RDM.

Syllable position and sonority are structurally related. Position should not
simply be controlled away as a nuisance variable. Users may instead run
separate analyses by syllable template or position when that comparison is
scientifically appropriate.

## License

Released under the MIT License; see [`LICENSE`](./LICENSE).
