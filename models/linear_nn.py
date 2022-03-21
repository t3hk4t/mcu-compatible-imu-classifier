import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import data_utils
from utils.analytics import SizeEstimator


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.name = 'Linear NN'
        self.linear1 = torch.nn.Linear(args.sensor_axis * args.buffer_size, 8)
        self.bn1 = torch.nn.BatchNorm1d(num_features=8)
        self.linear2 = torch.nn.Linear(8, 6)
        self.bn2 = torch.nn.BatchNorm1d(num_features=6)
        self.linear3 = torch.nn.Linear(6, 4)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.apply(self.init_weights)
        self.calculate_model_stats(args)

    def forward(self, x):
        x = self.linear1(x.view(x.shape[0], x.shape[1] * x.shape[2]))
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.linear3(x)
        if self.training:
            return x
        else:
            return F.softmax(x, dim=1)

    def forward_inference(self, x):
        output = self.forward(x)
        return F.softmax(output, dim=1)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
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
            print(
                f'Size of Parameters + input: {args.buffer_size * args.sensor_axis * 4 / 1024 + data_utils.count_parameters(model) * 4 / 1024}KB')



