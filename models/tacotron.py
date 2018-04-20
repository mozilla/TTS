# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
from TTS.utils.text.symbols import symbols
from TTS.layers.tacotron import Prenet, Encoder, Decoder, CBHG


class Tacotron(nn.Module):
    def __init__(self, embedding_dim=256, linear_dim=1025, mel_dim=80,
                 r=5, padding_idx=None):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(len(symbols), embedding_dim,
                                      padding_idx=padding_idx)
        print(" | > Number of characted : {}".format(len(symbols)))
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(256, mel_dim, r)
        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, characters, mel_specs=None, hiddens=None):
        B = characters.size(0)
        
        if hiddens is None:
            hiddens = self.init_rnn_hiddens(B)
        hiddens[0] = hiddens[0].transpose(0, 1).contiguous()
        hiddens[3] = hiddens[3].transpose(0, 1).contiguous()
        
        inputs = self.embedding(characters)
        # batch x time x dim
        hiddens[0] = self.encoder(inputs, hiddens[0])
        # batch x time x dim*r
        mel_outputs, alignments, hiddens[1], hiddens[2] =\
            self.decoder(hiddens[0], mel_specs, 
                         hiddens[1],
                         hiddens[2])
        # Reshape
        # batch x time x dim
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        hiddens[3] = self.postnet(mel_outputs, hiddens[3])
        linear_outputs = self.last_linear(hiddens[3])
        return mel_outputs, linear_outputs, alignments, hiddens
            
    def init_rnn_hiddens(self, B):
        weight = next(self.parameters()).data
        hiddens = [Variable(weight.new(B, 2, 128).zero_()),
                       Variable(weight.new(B, 256).zero_()),
                       [Variable(weight.new(B, 256).zero_()), 
                            Variable(weight.new(B, 256).zero_())],
                       Variable(weight.new(B, 2, self.mel_dim).zero_())]
        return hiddens