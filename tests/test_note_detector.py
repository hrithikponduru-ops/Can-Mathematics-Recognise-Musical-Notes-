import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import note_detector as nd

SR = 22050


def test_a4_is_440():
    assert nd.midi_to_freq(69) == pytest.approx(440.0)
    assert nd.freq_to_midi(440.0) == pytest.approx(69.0)


@pytest.mark.parametrize("name,midi", [("C4", 60), ("A4", 69), ("F#3", 54), ("Bb2", 46), ("E2", 40)])
def test_note_names(name, midi):
    assert nd.name_to_midi(name) == midi
    assert nd.name_to_midi(nd.midi_to_name(midi)) == midi


def test_naive_dft_matches_fft():
    x = np.random.default_rng(0).standard_normal(256)
    assert np.allclose(nd.naive_dft(x), np.fft.fft(x))


def test_parabolic_interpolation_beats_bin_spacing():
    f_true = 441.3
    t = np.arange(int(0.3 * SR)) / SR
    freqs, mag = nd.magnitude_spectrum(np.sin(2 * np.pi * f_true * t), SR, zero_pad=1)
    pf, _ = nd.find_spectral_peaks(freqs, mag)
    bin_width = freqs[1] - freqs[0]
    assert abs(pf[0] - f_true) < 0.1 * bin_width


@pytest.mark.parametrize("name", ["E2", "A2", "D3", "C4", "A4", "E5", "C6"])
def test_single_notes(name):
    x = nd.synth_note(nd.midi_to_freq(nd.name_to_midi(name)), 0.8, SR)
    segs, _, _ = nd.analyze(x, SR, max_notes=1)
    assert [n.name for n in segs[0].notes] == [name]


def test_weak_fundamental_like_a_low_guitar_string():
    # 2nd harmonic louder than the fundamental: a classic octave-error trap
    x = nd.synth_note(nd.midi_to_freq(40), 0.8, SR, harmonics=(0.5, 1.0, 0.6, 0.5, 0.3, 0.2))
    segs, _, _ = nd.analyze(x, SR, max_notes=1)
    assert segs[0].notes[0].name == "E2"


def test_out_of_tune_note_reports_cents():
    f = nd.midi_to_freq(69.2)            # A4, 20 cents sharp
    segs, _, _ = nd.analyze(nd.synth_note(f, 0.8, SR), SR, max_notes=1)
    assert segs[0].notes[0].name == "A4"
    assert segs[0].notes[0].cents == pytest.approx(20, abs=3)


@pytest.mark.parametrize("chord", [["C4", "E4", "G4"], ["A3", "C4", "E4"], ["E3", "B3"]])
def test_chords(chord):
    x = sum(nd.synth_note(nd.midi_to_freq(nd.name_to_midi(n)), 1.0, SR) for n in chord)
    segs, _, _ = nd.analyze(nd.normalise(x), SR, max_notes=len(chord) + 1)
    assert {n.name for n in segs[0].notes} == set(chord)


def test_demo_melody():
    sr, x = nd.demo_signal()
    segs, _, _ = nd.analyze(x, sr)
    got = [[n.name for n in s.notes] for s in segs]
    assert got == [sorted(n, key=nd.name_to_midi) for n, _ in nd.DEMO_EVENTS]
