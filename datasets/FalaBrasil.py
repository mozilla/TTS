# -*- coding: utf-8 -*-

import os
import numpy as np
import collections
import librosa
import torch
import fnmatch
import codecs
import itertools
import random

from torch.utils.data import Dataset

from TTS.utils.text import text_to_sequence
from TTS.utils.audio import AudioProcessor
from TTS.utils.data import (prepare_data, pad_per_step,
                            prepare_tensor, prepare_stop_target)

LAPSMAIL_TRANSCRIPTS = dict(zip(itertools.count(1), [
    'abrir email',
    'ler mensagens',
    'abrir caixa de entrada',
    'criar mensagem',
    'criar nova mensagem',
    'responder ao remetente',
    'responder',
    'responder a todos',
    'enviar',
    'salvar anexo',
    'adicionar contato',
    'inserir contato',
    'inserir destinatário',
    'anexar arquivo',
    'salvar mensagem',
    'salvar rascunho',
    'próxima',
    'próxima mensagem',
    'anterior',
    'mensagem anterior',
    'organizar por ordem alfabética',
    'organizar por data',
    'organizar por remetente',
    'exibir mensagens não lidas',
    'próxima não lida',
    'primeira mensagem',
    'primeira não lida',
    'última mensagem recebida',
    'última mensagem não lida',
    'não lidas',
    'mover mensagem',
    'encaminhar mensagem',
    'encaminhar',
    'excluir mensagem',
    'deletar',
    'enviar para lixeira',
    'fechar',
    'sair',
    'descartar',
    'procurar',
    'spam',
    'ler',
    'para',
    'gravar',
    'andrei',
    'anderson',
    'agnalda',
    'aldebaro',
    'bruno',
    'adalbere',
    'ana carolina',
    'fernanda',
    'nelson',
    'josué',
    'renan',
    'danilo',
    'jonatas',
    'lailson',
    'gustavo',
    'mariana',
    'muller',
    'diego',
    'diogo',
    'sílvia',
    'nagib',
    'marcos',
    'kelly',
    None,
    'mônica',
    None,
    'guilherme',
    'william',
    'jeferson',
    'claudomir',
    'lucila',
    'pedro',
    'rodrigo',
    'leonardo',
    'cláudio',
    'fabíola',
    'pelais',
    'rafael',
    'suânia',
    'charles',
    'marcel',
    'ericsson',
    'igor',
    'cleiton',
    'hugo',
    'hermerson',
]))


class FalaBrasilDataset(Dataset):

    def __init__(self, root_dir, outputs_per_step, sample_rate,
                text_cleaner, num_mels, min_level_db, frame_shift_ms,
                frame_length_ms, preemphasis, ref_level_db, num_freq, power,
                min_seq_len=0, is_for_val=False):

        self.frames = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, "*.txt"):
                trans_filename = os.path.join(root, filename)
                transcript = codecs.open(trans_filename, "r", "utf-8").read()
                wav_filename = os.path.splitext(trans_filename)[0]
                self.frames.append([wav_filename, transcript, transcript])

        self.root_dir = root_dir
        self.outputs_per_step = outputs_per_step
        self.sample_rate = sample_rate
        self.cleaners = text_cleaner
        self.min_seq_len = min_seq_len
        self.ap = AudioProcessor(sample_rate, num_mels, min_level_db, frame_shift_ms,
                                 frame_length_ms, preemphasis, ref_level_db, num_freq, power)
        print(" > Reading FalaBrasil from - {}".format(root_dir))
        print(" | > Number of instances : {}".format(len(self.frames)))
        self.is_for_val = is_for_val
        self._sort_frames()

    def load_wav(self, filename):
        try:
            audio = librosa.core.load(filename, sr=self.sample_rate)
            return audio
        except RuntimeError as e:
            print(" !! Cannot read file : {}".format(filename))

    def _sort_frames(self):
        r"""Sort sequences in ascending order"""
        lengths = np.array([len(ins[1]) for ins in self.frames])

        idxs = np.argsort(lengths)
        new_frames = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len or length > 500:
                ignored.append(idx)
            else:
                new_frames.append(self.frames[idx])
        print(" | > {} instances are ignored by min_seq_len ({})".format(len(ignored), self.min_seq_len))
        new_frames = np.array(new_frames)
        np.random.seed(4567)
        train, val = np.vsplit(new_frames[np.random.permutation(new_frames.shape[0])], (new_frames.shape[0]-100,))
        if self.is_for_val:
            self.frames = val
        else:
            self.frames = train

        print(" | > Used number of instances: {}".format(len(self.frames)))
        print(" | > Max length sequence {}".format(np.max(lengths)))
        print(" | > Min length sequence {}".format(np.min(lengths)))
        print(" | > Avg length sequence {}".format(np.mean(lengths)))


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir,
                                self.frames[idx][0]) + '.wav'
        text = self.frames[idx][1]
        text = np.asarray(text_to_sequence(text, [self.cleaners]), dtype=np.int32)
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)
        sample = {'text': text, 'wav': wav, 'item_idx': self.frames[idx][0]}
        return sample

    def get_dummy_data(self):
        r"""Get a dummy input for testing"""
        return torch.autograd.Variable(torch.ones(16, 143)).type(torch.LongTensor)

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
            keys = list()

            wav = [d['wav'] for d in batch]
            item_idxs = [d['item_idx'] for d in batch]
            text = [d['text'] for d in batch]

            text_lenghts = np.array([len(x) for x in text])
            max_text_len = np.max(text_lenghts)

            linear = [self.ap.spectrogram(w).astype('float32') for w in wav]
            mel = [self.ap.melspectrogram(w).astype('float32') for w in wav]
            mel_lengths = [m.shape[1] + 1 for m in mel]  # +1 for zero-frame

            # compute 'stop token' targets
            stop_targets = [np.array([0.]*(mel_len-1)) for mel_len in mel_lengths]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with largest length of the batch
            text = prepare_data(text).astype(np.int32)
            wav = prepare_data(wav)

            # PAD features with largest length + a zero frame
            linear = prepare_tensor(linear, self.outputs_per_step)
            mel = prepare_tensor(mel, self.outputs_per_step)
            assert mel.shape[2] == linear.shape[2]
            timesteps = mel.shape[2]

            # B x T x D
            linear = linear.transpose(0, 2, 1)
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            linear = torch.FloatTensor(linear)
            mel = torch.FloatTensor(mel)
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            return text, text_lenghts, linear, mel, mel_lengths, stop_targets, item_idxs[0]

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}"
                         .format(type(batch[0]))))
