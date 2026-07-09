"""Build frame tables from phraser, echoframe, and dutch_syllabifier."""

import pandas as pd
from dutch_syllabifier.sonority import sonority_weight

from sonority_rsa.data import prepare_frame_table

SKIP_REASONS = ['no_embedding', 'no_frames', 'no_syllable', 'no_sonority']


def build_frame_table(phrases, model_name, layers, collar=500, verbose=True):
    """
    Build a frame table of phone middle-frame vectors and sonority values.

    Loads one stored hidden-state embedding per phrase per layer and slices
    it to the middle frame of each phone. Phones are skipped (and counted)
    when the phrase has no stored embedding for a layer, no frames overlap
    the phone, the phone has no parent syllable, or its label has no
    sonority class.

    phrases: iterable of phraser Phrase objects with stored embeddings
    model_name: registered echoframe model name (e.g. 'wav2vec2')
    layers: list of hidden-state layers to extract
    collar: milliseconds of context stored around the phrase
    verbose: print a report of skipped phrases and phones
    """
    rows = []
    skipped = {reason: 0 for reason in SKIP_REASONS}

    for phrase in phrases:
        for layer in layers:
            embedding = _load_phrase_embedding(phrase, model_name, layer,
                collar)
            if embedding is None:
                skipped['no_embedding'] += 1
                continue
            rows.extend(_phone_rows(phrase, embedding, layer, skipped))

    if verbose:
        _report_skipped(skipped)
    if not rows:
        raise ValueError(
            'no phone frames extracted; are hidden states stored for these '
            f'phrases (model_name={model_name!r}, layers={layers})?')

    return prepare_frame_table(pd.DataFrame(rows))


def phone_sonority(phone):
    """
    Return the sonority weight for a phone, or None if it has no class.

    phone: phraser Phone (or any object with a .label attribute)
    """
    try:
        return sonority_weight(phone)
    except ValueError:
        return None


def _load_phrase_embedding(phrase, model_name, layer, collar):
    """
    Load one stored phrase embedding, or None when nothing is stored.

    phrase: phraser Phrase object
    model_name: registered echoframe model name
    layer: hidden-state layer
    collar: milliseconds of context stored around the phrase
    """
    try:
        return phrase.embedding(model_name, layer, collar=collar)
    except ValueError:
        return None


def _phone_rows(phrase, embedding, layer, skipped):
    """
    Yield one frame-table row per usable phone in a phrase.

    phrase: phraser Phrase object
    embedding: stored echoframe Embedding for the phrase
    layer: hidden-state layer being extracted
    skipped: counter dict updated in place
    """
    for phone in phrase.phones:
        syllable = phone.syllable
        if syllable is None:
            skipped['no_syllable'] += 1
            continue
        sonority = phone_sonority(phone)
        if sonority is None:
            skipped['no_sonority'] += 1
            continue
        try:
            sub = embedding.sub_embedding(phone, aggregate='middle')
        except ValueError:
            skipped['no_frames'] += 1
            continue
        yield {
            'syllable_id': syllable.key,
            'phone': phone.label,
            'sonority': sonority,
            'layer': layer,
            'vector': sub.data,
        }


def _report_skipped(skipped):
    """
    Print a one-line report of skip counts when anything was skipped.

    skipped: counter dict from build_frame_table
    """
    total = sum(skipped.values())
    if total == 0:
        return
    parts = [f'{reason}={count}' for reason, count in skipped.items()
        if count > 0]
    print(f'skipped {total} ({", ".join(parts)})')
