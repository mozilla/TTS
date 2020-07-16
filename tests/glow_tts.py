import os
import copy
import torch
import unittest
import numpy as np

from torch import optim
from torch import nn
from TTS.tts.utils.io import load_config
from TTS.tts.layers.losses import MSELossMasked
from TTS.tts.models.glow_tts import GlowTTS

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class GlowTTSTrainTest(unittest.TestCase):
    def test_train_step(self):
        input = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8, )).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, c.audio['num_mels'], 30 ).to(device)
        mel_postnet_spec = torch.rand(8, c.audio['num_mels'], 30 ).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        mel_lengths[-1] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = GlowTTS(num_chars=24,
                        hidden_channels=192,
                        filter_channels=768,
                        filter_channels_dp=256,
                        out_channels=80,
                        kernel_size=3,
                        num_heads=2,
                        num_layers_enc=6,
                        dropout_p=0.1,
                        num_blocks_dec=12,
                        kernel_size_dec=5,
                        dilation_rate=1,
                        num_block_layers=4,
                        dropout_p_dec=0.05,
                        num_speakers=0,
                        c_in_channels=0,
                        num_splits=4,
                        num_sqz=2,
                        sigmoid_scale=False,
                        rel_attn_winndow_size=4,
                        input_length=None,
                        mean_only=True,
                        hidden_channels_enc=192,
                        hidden_channels_dec=192,
                        prenet=True).to(device)

        model.train()
        # model_ref = copy.deepcopy(model)
        count = 0
        # for param, param_ref in zip(model.parameters(),
        #                             model_ref.parameters()):
        #     assert (param - param_ref).sum() == 0, param
        #     count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for i in range(5):
            (z, y_mean, y_log_scale, logdet), attn, o_dur_log, o_total_dur, o_mean, o_log_scale = model.forward(
                input, input_lengths, mel_spec, mel_lengths, gen=False)
            assert torch.sigmoid(stop_tokens).data.max() <= 1.0
            assert torch.sigmoid(stop_tokens).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(mel_postnet_out, mel_postnet_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        # for param, param_ref in zip(model.parameters(),
        #                             model_ref.parameters()):
        #     # ignore pre-higway layer since it works conditional
        #     # if count not in [145, 59]:
        #     assert (param != param_ref).any(
        #     ), "param {} with shape {} not updated!! \n{}\n{}".format(
        #         count, param.shape, param, param_ref)
        #     count += 1
