import torch
import torch.nn.functional as F
import sys

from utils.analytics import SizeEstimator
from utils import data_utils


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'Conv1D'
        self.conv1 = torch.nn.Conv1d(in_channels=args.sensor_axis, out_channels=32, kernel_size=6, padding=1, dilation=1, stride=4) # l_out = 50
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=12, kernel_size=7, padding=4, dilation=2, #l_out = 10
                                     stride=5)
        self.fc = torch.nn.Linear(in_features=12*10, out_features=args.output_size) # conv2_channels_out * l_out
        self.gn1 = torch.nn.GroupNorm(8, 32)
        self.gn2 = torch.nn.GroupNorm(4, 12)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.apply(self.init_weights)
        self.calculate_model_stats(args)

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

    def calculate_model_stats(self, args):
        if args.debug:
            model = self
            se = SizeEstimator(model, input_size=(args.sensor_axis, args.buffer_size))
            print(f'Model name: {self.name}')
            print(f'Parameter count: {data_utils.count_parameters(model)}')
            print(f'Parameter dtype: {se.get_parameter_dtype()}')
            print(f'Parameter size: {(data_utils.count_parameters(model) * 4) / 1024}KB')
            print(f'Size of Parameters + input: {args.buffer_size * args.sensor_axis * 4 / 1024 + data_utils.count_parameters(model) * 4 / 1024}KB')


