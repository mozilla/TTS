# coding: utf-8
import torch
from torch import nn
from math import sqrt
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG
from layers.gst_layers import GST
from utils.generic_utils import sequence_mask


class TacotronGST(nn.Module):
    def __init__(self,
                 num_chars,
                 r=5,
                 linear_dim=1025,
                 mel_dim=80,
                 memory_size=5,
                 attn_win=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True):
        super(TacotronGST, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(256)
        self.gst = GST(num_mel=80, num_heads=4, num_style_tokens=10, embedding_dim=256)
        self.decoder = Decoder(256, mel_dim, r, memory_size, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, separate_stopnet)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim),
            nn.Sigmoid())

    def forward(self, characters, text_lengths, mel_specs):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        gst_outputs = self.gst(mel_specs)
        gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
        encoder_outputs = encoder_outputs + gst_outputs
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