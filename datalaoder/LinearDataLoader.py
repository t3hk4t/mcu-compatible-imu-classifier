import torch
import numpy as np
import h5py
import itertools
from utils import data_utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, enabled_axis : list, overlap=100, data_buffer=200, axes=6, normalize=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.enabled_axis = [i for i, x in enumerate(enabled_axis) if x]

        if len(self.enabled_axis) == 0:
            raise ValueError('Dataset should contain at least one enabled axis')

        self.data_buffer = data_buffer * axes
        self.axes = axes
        overlap = overlap * 6

        with h5py.File(self.dataset_path, 'r') as hf:
            self.labels = list(hf.keys())

            data_list = []

            for label in self.labels:
                data = np.array(hf.get(label))
                data_list.append(np.array(list([data[i:i + self.data_buffer] for i in range(0, len(data), self.data_buffer - overlap)])[
                    :-2], dtype=float))

        data_list_y = []
        self.feature_lengths = []
        for idx, values in enumerate(data_list):
            y = [idx] * len(values)
            self.feature_lengths.append(len(values))
            if data_list_y:
                data_list_y = [*data_list_y, *y]
            else:
                data_list_y = [*y]

        self.Y = torch.Tensor(data_list_y).to(torch.int64)

        self.X_tmp = torch.Tensor(list(itertools.chain.from_iterable(data_list)))

        if normalize:
            self.normalize_data()
        else:
            new = torch.unsqueeze(self.X_tmp[:, self.enabled_axis[0]::6], 0)

            for value in range(1, len(self.enabled_axis)):
                new = torch.cat((new, torch.unsqueeze(self.X_tmp[:, self.enabled_axis[value]::6], 0)), 0)

            self.X = new.permute(1, 0, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_labels(self):
        return self.labels

    def get_feature_lengths(self):
        return np.array(self.feature_lengths)

    def normalize_data(self):
        normalized_data = data_utils.z_score(
            (self.X_tmp[:, 0::6], self.X_tmp[:, 1::6], self.X_tmp[:, 2::6], self.X_tmp[:, 3::6], self.X_tmp[:, 4::6], self.X_tmp[:, 5::6]),
            int(self.data_buffer / 6) * len(self.X))

        new = torch.unsqueeze(self.normalized_data[self.enabled_axis[0]], 0)

        for value in range(1, len(self.enabled_axis)):
            new = torch.cat((new, torch.unsqueeze(normalized_data[self.enabled_axis[value]], 0)), 0)

        self.X = new.permute(1, 0, 2)