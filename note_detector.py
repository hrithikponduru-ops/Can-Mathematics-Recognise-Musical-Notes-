#!/usr/bin/env python3
"""
note_detector.py - Can mathematics recognise musical notes?

Pipeline
  1. Load or record audio                        -> a list of numbers x[n]
  2. Find where each note starts                 -> short-time Fourier transform + spectral flux
  3. For each note: window, zero-pad, FFT        -> Fourier transform
  4. Find spectral peaks, refine with a parabola -> numerical analysis
  5. Group peaks into harmonic series            -> trigonometry / physics of vibrating strings
  6. Frequency -> note name + cents              -> logarithms

Usage
  python note_detector.py my_guitar.wav
  python note_detector.py --demo                      # synthetic test melody
  python note_detector.py --record 4                  # record 4 s from the microphone
  python note_detector.py my_piano.wav --max-notes 1  # monophonic (one note at a time)
  python note_detector.py my_piano.wav --save result.png
  python note_detector.py --verify-dft                # compare the textbook DFT with the FFT
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_HZ = 440.0
F_MIN, F_MAX = 27.5, 4200.0      # fundamental search range: A0 .. roughly C8
PEAK_F_MAX = 8000.0              # harmonics above F_MAX still help identify the fundamental


# ----------------------------------------------------------------------------
# 1. Notes and frequencies (equal temperament): f = 440 * 2^((m - 69) / 12)
# ----------------------------------------------------------------------------
def freq_to_midi(f):
    """Inverse of the equal-temperament formula: m = 69 + 12 log2(f / 440)."""
    return 69 + 12 * np.log2(np.asarray(f, dtype=float) / A4_HZ)


def midi_to_freq(m):
    return A4_HZ * 2.0 ** ((np.asarray(m, dtype=float) - 69) / 12)


def midi_to_name(m) -> str:
    m = int(round(float(m)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def name_to_midi(name: str) -> int:
    """'C4' -> 60, 'A4' -> 69, 'F#3' -> 54, 'Bb2' -> 46."""
    name = name.strip()
    letter, rest = name[0].upper(), name[1:]
    shift = 0
    while rest and rest[0] in "#b":
        shift += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + NOTE_NAMES.index(letter) + shift


# ----------------------------------------------------------------------------
# Audio input
# ----------------------------------------------------------------------------
def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()                       # remove DC offset (a 0 Hz "note")
    peak = np.max(np.abs(x)) if x.size else 0
    return x / peak if peak > 0 else x


def load_audio(path: str):
    """Returns (sample_rate, mono signal scaled to [-1, 1])."""
    if path.lower().endswith(".wav"):
        try:
            sr, x = wavfile.read(path)
            if x.dtype.kind == "i":
                x = x.astype(np.float64) / np.iinfo(x.dtype).max
            elif x.dtype.kind == "u":
                x = (x.astype(np.float64) - 128) / 128
            if x.ndim > 1:
                x = x.mean(axis=1)             # stereo -> mono
            return sr, normalise(x)
        except ValueError:
            pass                                # unusual WAV encoding: try soundfile
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("This file type needs the optional package: pip install soundfile\n"
                 "(or convert it: ffmpeg -i input.m4a output.wav)")
    x, sr = sf.read(path, always_2d=True)
    return sr, normalise(x.mean(axis=1))


def record(seconds: float, sr: int = 44100, save_to: str | None = None):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("Recording needs the optional package: pip install sounddevice")
    print(f"Recording for {seconds:.1f} s ... play now!")
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()
    x = normalise(x[:, 0])
    if save_to:
        wavfile.write(save_to, sr, (x * 32767).astype(np.int16))
        print(f"Saved recording to {save_to}")
    return sr, x


# ----------------------------------------------------------------------------
# Synthetic instrument: a note is a sum of sines at f, 2f, 3f, ... (Fourier series)
# ----------------------------------------------------------------------------
def synth_note(freq, dur, sr, harmonics=(1.0, 0.5, 0.33, 0.25, 0.2, 0.12, 0.08), decay=3.0):
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for k, amp in enumerate(harmonics, start=1):
        if k * freq < sr / 2:                   # stay below the Nyquist frequency
            y += amp * np.sin(2 * np.pi * k * freq * t)
    envelope = np.minimum(1.0, t / 0.005) * np.exp(-decay * t)   # pluck: fast attack, decay
    return y * envelope


DEMO_EVENTS = [(["C4"], 0.5), (["E4"], 0.5), (["G4"], 0.5), (["C5"], 0.5),
               (["A3"], 0.5), (["C4", "E4", "G4"], 1.2)]


def demo_signal(sr=22050, noise=0.01, seed=0):
    parts = [sum(synth_note(midi_to_freq(name_to_midi(n)), dur, sr) for n in notes)
             for notes, dur in DEMO_EVENTS]
    x = np.concatenate(parts)
    x = x + noise * np.random.default_rng(seed).standard_normal(len(x))
    return sr, normalise(x)


# ----------------------------------------------------------------------------
# The Fourier transform, from the definition (for learning) and via the FFT
# ----------------------------------------------------------------------------
def naive_dft(x: np.ndarray) -> np.ndarray:
    """X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N). Direct matrix form: O(N^2) operations."""
    n = np.arange(len(x))
    W = np.exp(-2j * np.pi * np.outer(n, n) / len(x))
    return W @ x


def verify_dft(N: int = 2048):
    x = np.random.default_rng(1).standard_normal(N)
    t0 = time.perf_counter(); X1 = naive_dft(x); t1 = time.perf_counter()
    X2 = np.fft.fft(x); t2 = time.perf_counter()
    err = np.max(np.abs(X1 - X2)) / np.max(np.abs(X2))
    print(f"N = {N}")
    print(f"  naive DFT : {1e3 * (t1 - t0):8.2f} ms   (~N^2 = {N**2:,} multiplications)")
    print(f"  FFT       : {1e3 * (t2 - t1):8.2f} ms   (~N log2 N = {int(N * np.log2(N)):,})")
    print(f"  max relative difference: {err:.2e}  (same answer, up to rounding error)")


def stft_magnitude(x, n_fft=2048, hop=512):
    """Short-time Fourier transform: FFT of overlapping windowed frames.
    Frame i is centred on time i*hop/sr. Returns |X| with shape (frames, n_fft//2+1)."""
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.abs(np.fft.rfft(xp[idx] * np.hanning(n_fft), axis=1))


def magnitude_spectrum(seg, sr, zero_pad=4):
    """Hann window -> zero-pad -> FFT. Scaled so a sine of amplitude A gives a peak of ~A."""
    win = np.hanning(len(seg))
    n_fft = int(2 ** np.ceil(np.log2(len(seg) * zero_pad)))
    mag = np.abs(np.fft.rfft(seg * win, n=n_fft)) * 2 / win.sum()
    return np.fft.rfftfreq(n_fft, 1 / sr), mag


# ----------------------------------------------------------------------------
# 2. Onset detection (where does each note start?)
# ----------------------------------------------------------------------------
def detect_onsets(x, sr, n_fft=2048, hop=512, delta=0.1, min_gap=0.08):
    """Spectral flux: how much new energy appears between consecutive frames."""
    S = stft_magnitude(x, n_fft, hop)
    L = np.log1p(1000 * S)                              # log compression, like the ear
    L = np.vstack([np.zeros((1, L.shape[1])), L])       # silence before the recording
    flux = np.maximum(0, np.diff(L, axis=0)).sum(axis=1)  # only count increases
    if len(flux) > 1:                        # first frame compares against pure silence,
        flux[0] = min(flux[0], flux[1:].max())   # so don't let it dwarf the real onsets
    flux /= flux.max() + 1e-12
    threshold = median_filter(flux, size=15, mode="constant") + delta   # adaptive threshold
    frames, _ = find_peaks(np.r_[0, flux], height=np.r_[1, threshold],   # leading 0 so a
                           distance=max(1, int(min_gap * sr / hop)))    # note at t=0 counts
    return (frames - 1) * hop / sr, flux, S


# ----------------------------------------------------------------------------
# 4. Spectral peaks with sub-bin precision
# ----------------------------------------------------------------------------
def parabolic_interpolation(y, k):
    """Fit a parabola through (k-1, y[k-1]), (k, y[k]), (k+1, y[k+1]).
    Returns the x-position and height of its vertex."""
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(k), b
    p = 0.5 * (a - c) / denom                # offset of the vertex, between -0.5 and 0.5
    return k + p, b - 0.25 * (a - c) * p


def find_spectral_peaks(freqs, mag, floor_db=-40.0, max_peaks=40):
    db = 20 * np.log10(mag + 1e-12)
    idx, _ = find_peaks(db, height=db.max() + floor_db, prominence=6)
    idx = idx[(freqs[idx] >= 0.9 * F_MIN) & (freqs[idx] <= PEAK_F_MAX)]
    df = freqs[1] - freqs[0]
    pf, pa = [], []
    for k in idx:
        kk, height = parabolic_interpolation(db, k)
        pf.append(kk * df)
        pa.append(10 ** (height / 20))
    pf, pa = np.array(pf), np.array(pa)
    order = np.argsort(pa)[::-1][:max_peaks]
    return pf[order], pa[order]


# ----------------------------------------------------------------------------
# 5. From peaks to notes: which fundamentals explain the harmonic series?
# ----------------------------------------------------------------------------
@dataclass
class Note:
    name: str
    freq: float
    midi: float
    cents: float          # how far from perfectly in tune (-50 .. +50)
    salience: float       # how strongly the spectrum supports this note
    n_harmonics: int


def match_harmonics(pf, pa, f0, max_h=10, tol_cents=40.0):
    """For each harmonic number h, the strongest peak within tol_cents of h*f0."""
    h = np.round(pf / f0)
    ok = (h >= 1) & (h <= max_h)
    cents = np.full(len(pf), np.inf)
    cents[ok] = 1200 * np.log2(pf[ok] / (h[ok] * f0))
    ok &= np.abs(cents) < tol_cents
    best = {}
    for i in np.flatnonzero(ok):
        hi = int(h[i])
        if hi not in best or pa[i] > pa[best[hi]]:
            best[hi] = i
    return best, ok


def score_candidate(pf, pa, f0):
    best, _ = match_harmonics(pf, pa, f0)
    if not best or (1 not in best and len(best) < 3):
        return None                       # need the fundamental, or a clear harmonic series
    hs = np.array(list(best.keys()), dtype=float)
    ii = np.array(list(best.values()))
    salience = np.sum(pa[ii] / hs)        # low harmonics count more
    w = pa[ii] / hs
    f0_refined = np.sum(w * pf[ii] / hs) / np.sum(w)   # weighted least-squares estimate
    return salience, f0_refined, len(best)


def estimate_notes(pf, pa, max_notes=3, rel_threshold=0.35):
    """Iterative harmonic-sum estimation: pick the best note, remove its harmonics, repeat."""
    active = np.ones(len(pf), dtype=bool)
    notes: list[Note] = []
    first = None
    for _ in range(max_notes):
        f, a = pf[active], pa[active]
        if len(f) == 0:
            break
        strongest = f[np.argsort(a)[::-1][:15]]
        cands = {round(fp / d, 2) for fp in strongest for d in (1, 2, 3, 4)
                 if F_MIN <= fp / d <= F_MAX}
        best = None
        for f0 in cands:
            r = score_candidate(f, a, f0)
            if r and (best is None or r[0] > best[0]):
                best = r
        if best is None:
            break
        salience, f0, nh = best
        if first is None:
            first = salience
        elif salience < rel_threshold * first:
            break
        m = float(freq_to_midi(f0))
        if all(midi_to_name(m) != n.name for n in notes):
            notes.append(Note(midi_to_name(m), f0, m, 100 * (m - round(m)), salience, nh))
        _, explained = match_harmonics(pf, pa, f0)
        active &= ~explained
    return sorted(notes, key=lambda n: n.freq)


# ----------------------------------------------------------------------------
# Full analysis
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    onset: float
    start: float
    end: float
    notes: list = field(default_factory=list)
    freqs: np.ndarray | None = None
    mag: np.ndarray | None = None
    peak_f: np.ndarray | None = None
    peak_a: np.ndarray | None = None


def analyze(x, sr, max_notes=3, rel_threshold=0.35, n_fft=2048, hop=512):
    onsets, flux, S = detect_onsets(x, sr, n_fft, hop)
    duration = len(x) / sr
    if len(onsets) == 0:
        onsets = np.array([0.0])
    bounds = list(onsets) + [duration]
    segments = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        start = t0 + 0.03                          # skip the noisy attack
        end = min(t1 - 0.01, t0 + 1.0)             # at most 1 s per note
        if end - start < 0.06:
            start = t0
        seg = x[int(start * sr):int(end * sr)]
        if len(seg) < 256:
            continue
        segments.append(Segment(t0, start, end))
        segments[-1].rms = float(np.sqrt(np.mean(seg ** 2)))
        freqs, mag = magnitude_spectrum(seg, sr)
        pf, pa = find_spectral_peaks(freqs, mag)
        s = segments[-1]
        s.freqs, s.mag, s.peak_f, s.peak_a = freqs, mag, pf, pa
    if segments:                                   # drop near-silent segments
        loudest = max(s.rms for s in segments)
        segments = [s for s in segments if s.rms > 0.05 * loudest]
    for s in segments:
        s.notes = estimate_notes(s.peak_f, s.peak_a, max_notes, rel_threshold)
    return segments, S, flux


def print_report(segments):
    print(f"\n{'#':>3}  {'time':>7}  {'note':<6} {'frequency':>10}  {'tuning':>9}  harmonics")
    print("-" * 56)
    for i, s in enumerate(segments, 1):
        for j, n in enumerate(s.notes):
            head = f"{i:>3}  {s.onset:6.2f}s" if j == 0 else " " * 13
            print(f"{head}  {n.name:<6} {n.freq:8.1f}Hz  {n.cents:+6.0f} ct  {n.n_harmonics}")
        if not s.notes:
            print(f"{i:>3}  {s.onset:6.2f}s  (no clear pitch)")
    seq = [("+".join(n.name for n in s.notes) or "?") for s in segments]
    print("\nSequence:", "  ".join(seq))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_results(x, sr, segments, S, n_fft=2048, hop=512, focus=None, save=None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    t = np.arange(len(x)) / sr

    # (a) waveform with detected notes
    ax = axes[0]
    ax.plot(t, x, lw=0.5, color="0.35")
    for s in segments:
        ax.axvline(s.onset, color="C3", lw=1, ls="--")
        label = "+".join(n.name for n in s.notes) or "?"
        ax.text(s.onset + 0.01, 0.97, label, transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color="C3", fontweight="bold")
    ax.set(title="Waveform x(t) and detected notes", xlabel="time (s)",
           ylabel="amplitude", xlim=(0, t[-1]))

    # (b) spectrogram with detected fundamentals
    ax = axes[1]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    keep = freqs >= 20
    times = np.arange(S.shape[0]) * hop / sr
    db = 20 * np.log10(S[:, keep].T + 1e-9)
    ax.pcolormesh(times, freqs[keep], db - db.max(), vmin=-80, vmax=0,
                  shading="auto", cmap="magma")
    for s in segments:
        for n in s.notes:
            ax.hlines(n.freq, s.onset, s.end, colors="cyan", lw=1.5)
    ax.set_yscale("log")
    ax.set_ylim(50, min(5000, sr / 2))
    ax.set(title="Spectrogram (short-time Fourier transform); cyan = detected fundamentals",
           xlabel="time (s)", ylabel="frequency (Hz)")

    # (c) spectrum of one segment with its harmonic series
    ax = axes[2]
    if segments:
        if focus is None:
            focus = int(np.argmax([len(s.notes) for s in segments]))
        s = segments[min(focus, len(segments) - 1)]
        ax.plot(s.freqs, 20 * np.log10(s.mag / s.mag.max() + 1e-12), lw=0.8, color="0.3")
        ax.plot(s.peak_f, 20 * np.log10(s.peak_a / s.mag.max()), "o", ms=3, color="C0",
                label="peaks (parabolic interpolation)")
        for i, n in enumerate(s.notes):
            col = f"C{i + 1}"
            for h in range(1, 11):
                if h * n.freq < PEAK_F_MAX:
                    ax.axvline(h * n.freq, color=col, lw=1, alpha=0.8 if h == 1 else 0.35,
                               ls="-" if h == 1 else ":")
            ax.plot([], [], color=col, label=f"{n.name} = {n.freq:.1f} Hz and harmonics")
        ax.set_xscale("log")
        ax.set_xlim(40, min(5000, sr / 2))
        ax.set_ylim(-80, 5)
        c_notes = [m for m in range(24, 109, 12) if 40 <= midi_to_freq(m) <= min(5000, sr / 2)]
        ax.set_xticks(midi_to_freq(c_notes))
        ax.set_xticklabels([f"{midi_to_name(m)}\n{midi_to_freq(m):.0f}" for m in c_notes])
        ax.set(title=f"Frequency spectrum |X(f)| of segment #{segments.index(s) + 1} "
                     f"(t = {s.start:.2f}-{s.end:.2f} s)",
               xlabel="frequency (Hz)", ylabel="level (dB)")
        ax.legend(loc="upper right", fontsize=8)

    if save:
        fig.savefig(save, dpi=130)
        print(f"Saved plot to {save}")
    else:
        plt.show()


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Estimate musical notes with Fourier analysis.")
    p.add_argument("audio", nargs="?", help="audio file (.wav; others need 'soundfile')")
    p.add_argument("--demo", action="store_true", help="analyse a synthetic melody")
    p.add_argument("--record", type=float, metavar="SEC", help="record from the microphone")
    p.add_argument("--save-audio", metavar="WAV", help="save recorded/demo audio")
    p.add_argument("--max-notes", type=int, default=3, help="max simultaneous notes (1 = melody)")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="extra notes must reach this fraction of the strongest (default 0.35)")
    p.add_argument("--segment", type=int, help="segment number to show in the spectrum plot")
    p.add_argument("--save", metavar="PNG", help="save the figure instead of showing it")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verify-dft", action="store_true", help="compare naive DFT with the FFT")
    args = p.parse_args(argv)

    if args.verify_dft:
        verify_dft()
        return
    if args.demo:
        sr, x = demo_signal()
        print("Demo melody played:", "  ".join("+".join(n) for n, _ in DEMO_EVENTS))
        if args.save_audio:
            wavfile.write(args.save_audio, sr, (x * 32767).astype(np.int16))
    elif args.record:
        sr, x = record(args.record, save_to=args.save_audio)
    elif args.audio:
        sr, x = load_audio(args.audio)
    else:
        p.error("give an audio file, --demo, --record SEC or --verify-dft")

    print(f"{len(x) / sr:.2f} s of audio at {sr} Hz")
    segments, S, _ = analyze(x, sr, args.max_notes, args.threshold)
    print_report(segments)
    if not args.no_plot:
        focus = args.segment - 1 if args.segment else None
        plot_results(x, sr, segments, S, focus=focus, save=args.save)


if __name__ == "__main__":
    main()
