import os
import librosa
import pickle
import copy
import numpy as np
from pprint import pprint
from scipy import signal

_mel_basis = None


class AudioProcessor(object):
    def __init__(self,
                 bits=None,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 num_freq=None):

        print(" > Setting up Audio Processor...")
        self.bits = bits
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.num_freq = num_freq
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        # librosa.output.write_wav(path, wav_norm.astype(np.int16), self.sample_rate)
        scipy.io.wavfile.write(path, self.sample_rate,
                               wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self,):
        n_fft = (self.num_freq - 1) * 2
        # fmin fmax from Tacotron2 paper
        return librosa.filters.mel(
            self.sample_rate, n_fft, n_mels=self.num_mels, fmin=125, fmax=7600)

    def _normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def _stft_parameters(self,):
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        print(
            " | > fft size: {}, hop length: {}, win length: {}".format(
                n_fft, hop_length, win_length
            )
        )
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D))
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S ** self.power))
        else:
            return self._griffin_lim(S ** self.power)

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        return self._normalize(S)

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x : x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    # WaveRNN repo specific functions
    # def mulaw_encode(self, wav, qc):
    #     mu = qc - 1
    #     wav_abs = np.minimum(np.abs(wav), 1.0)
    #     magnitude = np.log(1 + mu * wav_abs) / np.log(1. + mu)
    #     signal = np.sign(wav) * magnitude
    #     # Quantize signal to the specified number of levels.
    #     signal = (signal + 1) / 2 * mu + 0.5
    #     return signal.astype(np.int32)

    # def mulaw_decode(self, wav, qc):
    #     """Recovers waveform from quantized values."""
    #     mu = qc - 1
    #     # Map values back to [-1, 1].
    #     casted = wav.astype(np.float32)
    #     signal = 2 * (casted / mu) - 1
    #     # Perform inverse of mu-law transformation.
    #     magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
    #     return np.sign(signal) * magnitude

    def load_wav(self, filename, encode=False):
        x = librosa.load(filename, sr=self.sample_rate)[0]
        return x

    def encode_16bits(self, x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def quantize(self, x):
        return (x + 1.) * (2**self.bits - 1) / 2

    def dequantize(self, x):
        return 2 * x / (2**self.bits - 1) - 1
