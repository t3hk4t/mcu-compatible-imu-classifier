import torch
import torch.nn.functional as F
import sys

from utils.analytics import SizeEstimator
from utils import data_utils


class Model(torch.nn.Module):
    def __init__(self, config, in_channels, conv1_out_channels = 32, conv2_out_channels = 12):
        super().__init__()
        self.name = 'Conv1D'
        self.in_channels = in_channels
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=6, padding=1, dilation=1, stride=4) # l_out = 50
        self.conv2 = torch.nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=7, padding=4, dilation=2, #l_out = 10
                                     stride=5)
        self.fc = torch.nn.Linear(in_features=conv2_out_channels*10, out_features=config['output_size']) # conv2_channels_out * l_out
        self.gn1 = torch.nn.GroupNorm(8, conv1_out_channels)
        self.gn2 = torch.nn.GroupNorm(4, conv2_out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.gn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.gn2(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.shape[0], x.shape[1]*x.shape[2]))
        if not self.training:
            return F.softmax(x, dim=1)
        else:
            return x

    def forward_inference(self, x):

        output = self.forward(x)
        return F.softmax(output, dim=1)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def calculate_model_stats(self):
        return data_utils.count_parameters(self), 200 * self.in_channels * 4 / 1024 + data_utils.count_parameters(self) * 4 / 1024


