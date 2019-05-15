# coding: utf-8
import torch
from torch import nn
from math import sqrt
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG
from utils.generic_utils import sequence_mask


class Tacotron(nn.Module):

    def __init__(self,
                 num_chars,
                 c,
                 padding_idx=None):
        super(Tacotron, self).__init__()
        self.r = c.r
        self.mel_dim = c.audio['num_mels']
        self.linear_dim = c.audio['num_freq']
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(256)
        self.decoder = Decoder(256, self.mel_dim, c.r, c.memory_size,
                               c.windowing, c.attention_norm)
        self.postnet = PostCBHG(self.mel_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, self.linear_dim),
            nn.Sigmoid())

    def forward(self, characters, text_lengths, mel_specs):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
