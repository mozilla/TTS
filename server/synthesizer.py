import io
import sys
import os
import torch
import numpy as np
import yaml
import time
import librosa
import librosa.display

from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text import phonemes, symbols
#from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis, text_to_seqvec, trim_silence
from TTS.utils.speakers import load_speaker_mapping
from TTS.utils.visual import visualize

from parallel_wavegan.models import MelGANGenerator
from parallel_wavegan.utils.audio import AudioProcessor as AudioProcessorVocoder

import re
alphabets = r"([A-Za-z])"
prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = r"[.](com|net|org|io|gov)"


class Synthesizer(object):
    def __init__(self, config):
        # model paths
        TTS_MODEL = config.tts_checkpoint_file
        TTS_CONFIG = config.tts_config_file
        MELGAN_MODEL = config.melgan_checkpoint_file
        MELGAN_CONFIG = config.melgan_config_file

        # load TTS config
        TTS_CONFIG = load_config(TTS_CONFIG)
        self.TTS_CONFIG = TTS_CONFIG

        # load PWGAN config
        with open(MELGAN_CONFIG) as f:
            MELGAN_CONFIG = yaml.load(f, Loader=yaml.Loader)

        # Set some config fields manually for testing
        TTS_CONFIG.windowing = False
        TTS_CONFIG.use_forward_attn = True

        # Set the vocoder
        self.use_gl = False
        # NVIDIA GPU
        self.use_cuda = torch.cuda.is_available()

        # LOAD TTS MODEL
        # multi speaker
        self.speaker_id = None
        speakers = []

        # load the model
        num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
        self.model = setup_model(num_chars, len(speakers), TTS_CONFIG)

        # load the audio processor
        self.ap = AudioProcessor(**TTS_CONFIG.audio)

        # load model state
        cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

        # load the model
        self.model.load_state_dict(cp['model'])
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        # set model stepsize
        if 'r' in cp:
            self.model.decoder.set_r(cp['r'])

        # load PWGAN MelGAN
        self.vocoder_model = MelGANGenerator(**MELGAN_CONFIG["generator_params"])
        self.vocoder_model.load_state_dict(torch.load(MELGAN_MODEL, map_location="cpu")["model"]["generator"])
        self.vocoder_model.remove_weight_norm()
        self.ap_vocoder = AudioProcessorVocoder(**MELGAN_CONFIG['audio'])
        if self.use_cuda:
            self.vocoder_model.cuda()
        self.vocoder_model.eval();

    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    def split_into_sentences(self, text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
        text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if "\"" in text:
            text = text.replace(".\"", "\".")
        if "!" in text:
            text = text.replace("!\"", "\"!")
        if "?" in text:
            text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def tts(self, text):
        wav = None
        if False: # Split sentences
          wavs = []
          sens = self.split_into_sentences(text)
          print(sens)
          if not sens:
              sens = [text+'.']
          for sen in sens:
              # preprocess the given text
              inputs = text_to_seqvec(sen, self.TTS_CONFIG, self.use_cuda)
              # synthesize voice
              wav_sen = self.tts_melgan(text)
              # trim silence
              #wav_sen = trim_silence(wav_sen, self.ap)

              wavs += list(wav_sen)
              wavs += [0] * 10000
          wav = wavs
        else:
          wav = self.tts_melgan(text)

        out = io.BytesIO()
        self.save_wav(wav, out);
        return out

    def tts_melgan(self, text, figures=True):
        t_1 = time.time()
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(self.model, text, self.TTS_CONFIG, self.use_cuda, self.ap, self.speaker_id, style_wav=None, truncated=False, enable_eos_bos_chars=self.TTS_CONFIG.enable_eos_bos_chars)
        if self.TTS_CONFIG.model == "Tacotron" and not self.use_gl:
            mel_postnet_spec = self.ap.out_linear_to_mel(mel_postnet_spec.T).T
        mel_postnet_spec = self.ap._denormalize(mel_postnet_spec)
        print(mel_postnet_spec.shape)
        print("max- ", mel_postnet_spec.max(), " -- min- ", mel_postnet_spec.min())
        if not self.use_gl:
            waveform = self.vocoder_model.inference(torch.FloatTensor(self.ap_vocoder._normalize(mel_postnet_spec).T).unsqueeze(0), hop_size=self.ap_vocoder.hop_length)
        if self.use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.numpy()
        print(waveform.shape)
        print(" >  Run-time: {}".format(time.time() - t_1))
        if figures:
            visualize(alignment, mel_postnet_spec, stop_tokens, text, self.ap.hop_length, self.TTS_CONFIG, self.ap._denormalize(mel_spec))
        return waveform
