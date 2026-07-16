"""Real phraser segments and a real echoframe store for the tests.

The corpus fixture builds a temporary phraser LMDB store with linked
Phrase > Syllable > Phone segments and a temporary echoframe store with
hidden-state payloads for two layers, so the tests exercise the real
key derivation, frame timing, and middle-frame slicing.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest
import soundfile
from echoframe import EchoframeMetadata
from echoframe import Store as EchoframeStore
from phraser import Store as PhraserStore
from phraser.models import Audio, Phone, Phrase, Syllable

MODEL_NAME = 'wav2vec2'
LAYERS = [0, 1]
COLLAR = 500
SOURCE_ID = 'test-source'
N_FRAMES = 20
DIM = 3


def payload(phrase_index, layer):
    """The stored hidden-state matrix for one phrase and layer."""
    rng = np.random.default_rng(10 * phrase_index + layer)
    return rng.normal(size=(N_FRAMES, DIM))


class Corpus:
    """Two stored phrases with syllables, plus unusable edge cases."""

    def __init__(self, root):
        with redirect_stdout(io.StringIO()):
            self.phraser_store = PhraserStore(
                path=str(root / 'phraser_lmdb'))
            self.echoframe_store = EchoframeStore(root / 'echoframe')
        self.audio_filename = root / 'test.wav'
        self._write_audio()
        self.echoframe_store.register_model(MODEL_NAME)
        self.echoframe_store.attach_phraser_store(SOURCE_ID,
            self.phraser_store)
        self._build_segments()
        self._store_payloads()

    def _build_segments(self):
        # Syllable spans never overlap: phraser resolves children by a
        # time scan over the audio, so overlapping syllables would share
        # phones. Phones are placed so their overlapping 20ms/25ms frames
        # and the selected middle frame are exact: a 0-100ms phone
        # overlaps payload rows 0-4 (middle row 2), a 100-200ms phone
        # rows 4-9 (middle row 6), a 200-300ms phone rows 9-14 (middle
        # row 11), and a 300-400ms phone rows 14-19 (middle row 16).
        self.audio = self.phraser_store.create(Audio,
            filename=str(self.audio_filename), sample_rate=16000,
            duration=3000, n_channels=1, save=False)
        self.segments = []
        self.ph1 = self._phrase('ph1', 0, 400)
        self.ph2 = self._phrase('ph2', 400, 800)
        self.ph3 = self._phrase('ph3', 800, 1200)
        self.s1 = self._syllable(self.ph1, 0, 200,
            [('p', 0, 100), ('a', 100, 200)])
        self.s2 = self._syllable(self.ph1, 200, 400,
            [('s', 200, 300), ('t', 300, 400)])
        self.s3 = self._syllable(self.ph2, 400, 600,
            [('m', 400, 500), ('l', 500, 600)])
        self.syllables = [self.s1, self.s2, self.s3]

        # A separate 90-syllable population for analyses that enforce the
        # minimum subset size. Short, non-overlapping phones share the same
        # stored phrase payload but retain distinct syllable keys.
        self.ph4 = self._phrase('ph4', 2000, 2400)
        self.analysis_syllables = []
        for index in range(90):
            start = 2000 + 4 * index
            label = 'a' if index % 2 else 'p'
            self.analysis_syllables.append(self._syllable(
                self.ph4, start, start + 4, [(label, start, start + 4)]))

        # edge cases: no stored payload for ph3, phones far outside ph1's
        # stored payload (no frames overlap), an unknown label, and a
        # syllable never linked to a phrase
        self.s_unstored = self._syllable(self.ph3, 800, 900,
            [('a', 800, 900)])
        self.s_bad = self._syllable(self.ph1, 1300, 1500,
            [('(..)', 1300, 1400), ('a', 1400, 1500)])
        self.s_unlinked = self._syllable(None, 1600, 1700,
            [('a', 1600, 1700)])

        for segment in self.segments:
            segment.add_audio(self.audio, update_database=False,
                propagate=False)
        self.phraser_store.save_many(self.segments)

    def _write_audio(self):
        """Write varying-amplitude speech-like audio for intensity tests."""
        sample_rate = 16000
        time = np.arange(3 * sample_rate) / sample_rate
        envelope = 0.1 + 0.15 * time / time[-1]
        signal = envelope * np.sin(2 * np.pi * 220 * time)
        soundfile.write(self.audio_filename, signal, sample_rate,
            subtype='FLOAT')

    def _store_payloads(self):
        for phrase_index, phrase in enumerate([self.ph1, self.ph2, self.ph4]):
            for layer in LAYERS:
                key = self.echoframe_store.make_echoframe_key(
                    'hidden_state', model_name=MODEL_NAME,
                    phraser_key=phrase.key, collar=COLLAR, layer=layer)
                metadata = EchoframeMetadata(key,
                    store=self.echoframe_store, model_name=MODEL_NAME,
                    phraser_source_id=SOURCE_ID)
                self.echoframe_store.save(key, metadata,
                    payload(phrase_index, layer))

    def _phrase(self, label, start, end):
        phrase = self.phraser_store.create(Phrase, label=label, start=start,
            end=end, save=False)
        self.segments.append(phrase)
        return phrase

    def _syllable(self, phrase, start, end, phones):
        syllable = self.phraser_store.create(Syllable, label='syl',
            start=start, end=end, save=False)
        if phrase is not None:
            syllable._add_phrase(phrase, update_database=False)
        self.segments.append(syllable)
        for label, phone_start, phone_end in phones:
            self._phone(syllable, label, phone_start, phone_end)
        return syllable

    def _phone(self, syllable, label, start, end):
        phone = self.phraser_store.create(Phone, label=label, start=start,
            end=end, save=False)
        phone.add_parent(syllable)
        self.segments.append(phone)
        return phone


@pytest.fixture(scope='session')
def corpus(tmp_path_factory):
    return Corpus(tmp_path_factory.mktemp('stores'))
