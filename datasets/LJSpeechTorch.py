import os
import numpy as np
import collections
import librosa
import torch
from torch.utils.data import Dataset

from TTS.utils.text import text_to_sequence
from TTS.utils.audio import AudioProcessor
from TTS.utils.data import (prepare_data, pad_per_step,
                            prepare_tensor, prepare_stop_target)


class LJSpeechDataset(Dataset):

    def __init__(self, csv_file, root_dir, ap, outputs_per_step,
                 text_cleaner, min_seq_len=0):

        with open(csv_file, "r") as f:
            self.frames = [line.split('|') for line in f]
        self.root_dir = root_dir
        self.outputs_per_step = outputs_per_step
        self.cleaners = text_cleaner
        self.min_seq_len = min_seq_len
        self.ap = ap
        print(" > Reading LJSpeech from - {}".format(root_dir))
        print(" | > Number of instances : {}".format(len(self.frames)))
        self.items = [None] * len(self.frames)
        self._sort_frames()

    def _sort_frames(self):
        r"""Sort instances by text length in ascending order"""
        lengths = np.array([len(ins[1]) for ins in self.frames])

        print(" | > Max length sequence {}".format(np.max(lengths)))
        print(" | > Min length sequence {}".format(np.min(lengths)))
        print(" | > Avg length sequence {}".format(np.mean(lengths)))

        idxs = np.argsort(lengths)
        new_frames = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len:
                ignored.append(idx)
            else:
                new_frames.append(self.frames[idx])
        print(" | > {} instances are ignored by min_seq_len ({})".format(
            len(ignored), self.min_seq_len))
        self.frames = new_frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if self.items[idx] is None:
            wav_name = os.path.join(self.root_dir,
                                    self.frames[idx][0]) + '.wav'
            text = self.frames[idx][1]
            text = torch.IntTensor(text_to_sequence(text, [self.cleaners]))
            wav = self.ap.load_wav(wav_name)
            wav = wav.unsqueeze(0)
            spec = self.ap.spectrogram(wav)
            mel_spec = self.ap.linear_to_mel(spec)
            spec_len = spec.shape[-1]
            sample = {'text': text, 'spec': spec, 'mel': mel_spec,
                    'item_idx': self.frames[idx][0], 'spec_len': spec_len}
            self.items[idx] = sample
        else:
            sample = self.items[idx]
        return sample

    def get_dummy_data(self):
        r"""Get a dummy input for testing"""
        return torch.ones(16, 143).type(torch.LongTensor)

    def collate_fn(self, batch):
        r"""
            Perform preprocessing and create a final data batch:
            1. PAD sequences with the longest sequence in the batch
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences that can be divided by r.
            4. Convert Numpy to Torch tensors.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            B = len(batch)
            M = batch[0]['mel'].shape[1]
            S = batch[0]['spec'].shape[1]
            item_idxs = [d['item_idx'] for d in batch]

            text_lens = torch.LongTensor([len(d['text']) for d in batch])
            max_text_len = text_lens.max()

            spec_lens = torch.LongTensor([d['spec_len']+1 for d in batch])
            max_spec_len = spec_lens.max()
            remainder = max_spec_len % self.outputs_per_step
            if remainder > 0:
                pad_len = (self.outputs_per_step - remainder)
            else:
                pad_len = 0

            assert (max_spec_len + pad_len) % self.outputs_per_step == 0

            text = torch.LongTensor(B, max_text_len)
            mel = torch.FloatTensor(B, M, max_spec_len + pad_len)
            spec = torch.FloatTensor(B, S, max_spec_len + pad_len)
            stop_targets = torch.FloatTensor(B, max_spec_len + pad_len)
            text.zero_()
            mel.zero_()
            spec.zero_()
            stop_targets.zero_()
            for idx, d in enumerate(batch):
                t = d['text']
                s = d['spec']
                m = d['mel']
                text[idx, :t.shape[0]] = t
                spec[idx, :, :s.shape[-1]] = s
                mel[idx, :, :m.shape[-1]] = m
                stop_targets[idx, m.shape[-1]:] = 1.0
            spec.transpose_(1, 2)
            mel.transpose_(1, 2)
            return text, text_lens, spec, mel, spec_lens, stop_targets, item_idxs[0]
        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}"
                         .format(type(batch[0]))))
