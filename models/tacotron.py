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
        
        self.encoder_rnn_hidden = None
        self.attention_rnn_hidden = None
        self.decoder_rnn_hidden = None
        self.postnet_rnn_hidden = None

    def forward(self, characters, mel_specs=None, start=True):
        B = characters.size(0)
        inputs = self.embedding(characters)
        self.init_rnn_hidden_states(inputs, start)
        # batch x time x dim
        self.encoder_rnn_hidden = self.encoder(inputs, self.encoder_rnn_hidden)
        # batch x time x dim*r
        mel_outputs, alignments, self.attention_rnn_hidden, self.decoder_rnn_hiddens =\
            self.decoder(self.encoder_rnn_hidden, mel_specs, 
                         self.attention_rnn_hidden,
                         self.decoder_rnn_hiddens)
        # Reshape
        # batch x time x dim
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs, self.postnet_rnn_hidden = self.postnet(mel_outputs, self.postnet_rnn_hidden)
        linear_outputs = self.last_linear(linear_outputs)
        
        return mel_outputs, linear_outputs, alignments
            
    def init_rnn_hidden_states(self, inputs, start):
        B = inputs.size(0)
        if start:
            self.encoder_rnn_hidden = Variable(inputs.data.new(2, B, 128).zero_())
            self.attention_rnn_hidden = Variable(inputs.data.new(B, 256).zero_())
            self.decoder_rnn_hiddens = [Variable(inputs.data.new(B, 256).zero_()), 
                                        Variable(inputs.data.new(B, 256).zero_())]
            self.postnet_rnn_hidden = Variable(inputs.data.new(2, B, self.mel_dim).zero_())
        else:
            self.encoder_rnn_hidden = self.encoder_rnn_hidden.detach()
            self.attention_rnn_hidden = self.attention_rnn_hidden.detach()
            self.decoder_rnn_hiddens = [h.detach() for h in self.decoder_rnn_hiddens]
            self.postnet_rnn_hidden = self.postnet_rnn_hidden.detach()