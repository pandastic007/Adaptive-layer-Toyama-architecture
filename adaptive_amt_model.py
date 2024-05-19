import torch
import torch.nn as nn
from model.model_spec2midi import Encoder_SPEC2MIDI, Decoder_SPEC2MIDI

class AdaptiveAMTModel(nn.Module):
    def __init__(self, encoder, decoder, num_piano_types, config):
        super(AdaptiveAMTModel, self).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList([decoder for _ in range(num_piano_types)])
        self.num_piano_types = num_piano_types
        self.config = config

    def forward(self, x, piano_type):
        encoded = self.encoder(x)
        outputs = [self.decoders[i](encoded) for i in range(self.num_piano_types)]
        return outputs[piano_type]

