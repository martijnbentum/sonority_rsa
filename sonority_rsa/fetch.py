"""Fetch syllable populations from phraser and echoframe stores."""

import numpy as np
from dutch_syllabifier.sonority import sonority_weight

SKIP_REASONS = ['no_phrase', 'no_embedding', 'no_sonority', 'no_frames',
    'constant_vector', 'no_phones']


class SyllableData:
    """Middle-frame vectors and sonority values for one syllable."""

    def __init__(self, key, phone_labels, sonority, vectors,
            phone_segments=None):
        """
        key: phraser syllable key
        phone_labels: phone label per row
        sonority: sonority weight per row
        vectors: middle-frame hidden-state vector per row
        phone_segments: source phraser Phone per row, when available
        """
        self.key = key
        self.phone_labels = list(phone_labels)
        self.sonority = np.asarray(sonority, dtype=float)
        self.vectors = np.asarray(vectors, dtype=float)
        self.phone_segments = (list(phone_segments)
            if phone_segments is not None else [])
        self.intensity = None

    def __repr__(self):
        return (f'SyllableData(key={self.key!r}, '
            f'phones={" ".join(self.phone_labels)})')

    def __len__(self):
        return len(self.phone_labels)


class SyllablePopulation:
    """All fetched syllables for one model layer."""

    def __init__(self, model_name, layer, collar, syllables, skipped):
        """
        model_name: registered echoframe model name
        layer: hidden-state layer
        collar: milliseconds of context stored around the phrase
        syllables: list of SyllableData
        skipped: skip counts by reason
        """
        self.model_name = model_name
        self.layer = layer
        self.collar = collar
        self.syllables = syllables
        self.skipped = skipped

    def __repr__(self):
        return (f'SyllablePopulation(model={self.model_name!r}, '
            f'layer={self.layer}, n_syllables={len(self)})')

    def __len__(self):
        return len(self.syllables)

    def __getitem__(self, index):
        return self.syllables[index]

    def __iter__(self):
        return iter(self.syllables)

    @property
    def keys(self):
        """Syllable keys in population order."""
        return [syllable.key for syllable in self.syllables]


def fetch_syllable_data(syllables, model_name, layer, echoframe_store,
        collar=500, verbose=True):
    """
    Fetch middle-frame vectors and sonority for phraser syllables.

    Loads one stored hidden-state embedding per phrase and slices it to
    the middle frame of each phone. Syllables are skipped (and counted)
    when they are not linked to a phrase, the phrase has no stored
    embedding, or none of their phones is usable; phones are skipped when
    their label has no sonority class, no frames overlap them, or their
    stored vector is constant across features (a broken/zeroed embedding,
    whose correlation distance would be undefined).

    Raises ValueError when no syllable is usable, or when every fetched
    phone shares one sonority class (RSA needs sonority variation). Under
    run_analysis the latter drops just that layer rather than aborting.

    syllables: list of phraser Syllable objects with linked phones
    model_name: registered echoframe model name (e.g. 'wav2vec2')
    layer: hidden-state layer to fetch
    echoframe_store: echoframe Store holding the hidden states
    collar: milliseconds of context stored around the phrase
    verbose: print a report of skip counts
    """
    skipped = {reason: 0 for reason in SKIP_REASONS}
    data = []

    for phrase_key, phrase_syllables in _group_by_phrase(syllables,
            skipped).items():
        try:
            embedding = echoframe_store.phraser_key_to_embedding(phrase_key,
                model_name, layer, collar=collar)
        except ValueError:
            skipped['no_embedding'] += len(phrase_syllables)
            continue
        for syllable in phrase_syllables:
            syllable_data = _syllable_data(syllable, embedding, skipped)
            if syllable_data is not None:
                data.append(syllable_data)

    if verbose:
        _report_skipped(skipped)
    if not data:
        raise ValueError(
            'no syllables fetched; are hidden states stored for these '
            f'phrases (model_name={model_name!r}, layer={layer})?')

    sonority_classes = np.unique(np.concatenate([s.sonority for s in data]))
    if sonority_classes.size < 2:
        raise ValueError(
            'all fetched phones share one sonority class '
            f'(model_name={model_name!r}, layer={layer}); RSA needs '
            'sonority variation across the population')

    return SyllablePopulation(model_name, layer, collar, data, skipped)


def phone_sonority(phone):
    """
    Return the sonority weight for a phone, or None if it has no class.

    phone: phraser Phone (or any object with a .label attribute)
    """
    try:
        return sonority_weight(phone)
    except ValueError:
        return None


def _group_by_phrase(syllables, skipped):
    """
    Group syllables by their phrase key, preserving input order.

    syllables: list of phraser Syllable objects
    skipped: counter dict updated in place
    """
    groups = {}
    for syllable in syllables:
        phrase_key = syllable.phrase_key
        if phrase_key is None:
            skipped['no_phrase'] += 1
            continue
        groups.setdefault(phrase_key, []).append(syllable)
    return groups


def _syllable_data(syllable, embedding, skipped):
    """
    Build SyllableData for one syllable, or None if no phone is usable.

    syllable: phraser Syllable object
    embedding: stored echoframe Embedding for the syllable's phrase
    skipped: counter dict updated in place
    """
    phone_labels, sonority, vectors, phone_segments = [], [], [], []
    for phone in syllable.phones:
        weight = phone_sonority(phone)
        if weight is None:
            skipped['no_sonority'] += 1
            continue
        try:
            sub = embedding.sub_embedding(phone, aggregate='middle')
        except ValueError:
            skipped['no_frames'] += 1
            continue
        vector = np.asarray(sub.data, dtype=float)
        if vector.size and np.ptp(vector) == 0:
            # a vector that is constant across features has zero variance,
            # which makes its correlation distance to every other phone
            # undefined (NaN); a real middle frame is never constant, so
            # this flags a broken/zeroed embedding rather than valid data
            skipped['constant_vector'] += 1
            continue
        phone_labels.append(phone.label)
        sonority.append(weight)
        vectors.append(vector)
        phone_segments.append(phone)
    if not phone_labels:
        skipped['no_phones'] += 1
        return None
    return SyllableData(syllable.key, phone_labels, sonority, vectors,
        phone_segments=phone_segments)


def _report_skipped(skipped):
    """
    Print a one-line report of skip counts when anything was skipped.

    skipped: counter dict from fetch_syllable_data
    """
    total = sum(skipped.values())
    if total == 0:
        return
    parts = [f'{reason}={count}' for reason, count in skipped.items()
        if count > 0]
    print(f'skipped {total} ({", ".join(parts)})')
