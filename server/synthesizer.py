import io
import os
import librosa
import torch
import scipy
import numpy as np
import soundfile as sf
from utils.text import text_to_sequence, phoneme_to_sequence
from utils.generic_utils import load_config, setup_model
from utils.audio import AudioProcessor
from utils.text.symbols import phonemes, symbols
from models.tacotron import Tacotron
from matplotlib import pylab as plt


class Synthesizer(object):
    def load_model(self, model_path, model_name, model_config, use_cuda):
        model_config = os.path.join(model_path, model_config)
        self.model_file = os.path.join(model_path, model_name)
        print(" > Loading model ...")
        print(" | > model config: ", model_config)
        print(" | > model file: ", self.model_file)
        config = load_config(model_config)
        self.config = config
        self.use_cuda = use_cuda
        self.ap = AudioProcessor(**config.audio)
        num_chars = len(phonemes) if config.use_phonemes else len(symbols)
        self.model = setup_model(num_chars, config)
        # load model state
        if use_cuda:
            cp = torch.load(self.model_file)
        else:
            cp = torch.load(
                self.model_file, map_location=lambda storage, loc: storage)
        # load the model
        self.model.load_state_dict(cp['model'])
        if use_cuda:
            self.model.cuda()
        self.model.eval()

    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    def tts(self, text):
        text_cleaner = [self.config.text_cleaner]
        wavs = []
        for sen in text.split('.'):
            if len(sen) < 3:
                continue
            sen = sen.strip()
            sen += '.'
            print(sen)
            sen = sen.strip()
            if self.config.use_phonemes:
                seq = np.asarray(
                    phoneme_to_sequence(text, text_cleaner, self.config.phoneme_language, self.config.enable_eos_bos_chars),
                    dtype=np.int32)
            else:
                seq = np.asarray(text_to_sequence(text, text_cleaner), dtype=np.int32)
            chars_var = torch.from_numpy(seq).unsqueeze(0).long()
            if self.use_cuda:
                chars_var = chars_var.cuda()
            decoder_output, postnet_output, alignments, stop_tokens = self.model.inference(
                chars_var)
            postnet_output = postnet_output[0].data.cpu().numpy()
            if self.config.model == "Tacotron":
                wav = self.ap.inv_spectrogram(postnet_output.T)
            else:
                wav = self.ap.inv_mel_spectrogram(postnet_output.T)           
            out = io.BytesIO()
            wavs += list(wav)
            wavs += [0] * 10000
        self.save_wav(wavs, out)
        return out