import torch
import numpy as np
import h5py
from utils import data_utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, overlap=100, data_buffer=200, axes=6, normalize=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.data_buffer = data_buffer * axes
        self.axes = axes
        overlap = overlap * 6
        self.gyro_std = 0
        self.gyro_mean = 0
        self.accel_std = 0
        self.accel_mean = 0

        with h5py.File(self.dataset_path, 'r') as hf:
            self.labels = list(hf.keys())
            drive_off = np.array(hf.get('drive_off'))
            drive_on = np.array(hf.get('drive_on'))
            stand_off = np.array(hf.get('stand_off'))
            stand_on = np.array(hf.get('stand_on'))

        drive_off_x = np.array(
            list([drive_off[i:i + self.data_buffer] for i in range(0, len(drive_off), self.data_buffer - overlap)])[
            :-2], dtype=float)
        drive_on_x = np.array(
            list([drive_on[i:i + self.data_buffer] for i in range(0, len(drive_on), self.data_buffer - overlap)])[:-2],
            dtype=float)
        stand_off_x = np.array(
            list([stand_off[i:i + self.data_buffer] for i in range(0, len(stand_off), self.data_buffer - overlap)])[
            :-2], dtype=float)
        stand_on_x = np.array(
            list([stand_on[i:i + self.data_buffer] for i in range(0, len(stand_on), self.data_buffer - overlap)])[:-2],
            dtype=float)

        drive_off_y = np.zeros(len(drive_off_x), dtype=int)
        drive_on_y = np.zeros(len(drive_on_x), dtype=int)
        drive_on_y[:] = 1
        stand_off_y = np.zeros(len(stand_off_x), dtype=int)
        stand_off_y[:] = 2
        stand_on_y = np.zeros(len(stand_on_x), dtype=int)
        stand_on_y[:] = 3

        drive_off_y = torch.from_numpy(drive_off_y).to(torch.int64)
        drive_on_y = torch.from_numpy(drive_on_y).to(torch.int64)
        stand_off_y = torch.from_numpy(stand_off_y).to(torch.int64)
        stand_on_y = torch.from_numpy(stand_on_y).to(torch.int64)

        self.Y = torch.cat((drive_off_y, drive_on_y, stand_off_y, stand_on_y), dim=0)
        self.X_tmp = torch.cat((torch.from_numpy(drive_off_x), torch.from_numpy(drive_on_x), torch.from_numpy(stand_off_x),
                            torch.from_numpy(stand_on_x)), dim=0)

        new = torch.cat((torch.unsqueeze(self.X_tmp[:, 0::6], 0), torch.unsqueeze(self.X_tmp[:, 1::6], 0),
                         torch.unsqueeze(self.X_tmp[:, 2::6], 0),
                         torch.unsqueeze(self.X_tmp[:, 3::6], 0), torch.unsqueeze(self.X_tmp[:, 4::6], 0),
                         torch.unsqueeze(self.X_tmp[:, 5::6], 0)), 0)
        # C B F
        self.X = new.permute(1, 0, 2)

        if normalize:
            self.normalize_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def normalize_data(self):
        normalized_data = data_utils.z_score(
            (self.X_tmp[:, 0::6], self.X_tmp[:, 1::6], self.X_tmp[:, 2::6], self.X_tmp[:, 3::6], self.X_tmp[:, 4::6], self.X_tmp[:, 5::6]),
            int(self.data_buffer / 6) * len(self.X))

        new = torch.cat((torch.unsqueeze(normalized_data[0], 0), torch.unsqueeze(normalized_data[1], 0),
                         torch.unsqueeze(normalized_data[2], 0),
                         torch.unsqueeze(normalized_data[3], 0), torch.unsqueeze(normalized_data[4], 0),
                         torch.unsqueeze(normalized_data[5], 0)), 0)
        self.X = new.permute(1, 0, 2)