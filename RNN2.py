import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AudioHandler import AudioHandler
import os

seq_length = 22050
BATCH_SIZE = 64
BUFFER_SIZE = 10000

audio_arrays = AudioHandler.get_audio_arrays("AudioDataset", normalized=True)

class AudioRNN(nn.Module):

    def __init__(self, i_size, h1_size, h2_size, o_size):
        super(AudioRNN, self).__init__()

        # Linear means to apply a linear transformation to incoming data (aka applies weights and an optional bias)
        self.h1 = nn.Linear(i_size, h1_size)
        self.h2 = nn.Linear(h1_size, h2_size)
