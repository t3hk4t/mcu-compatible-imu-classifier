from preprocessor.dataset_loader import DataLoader
import h5py
import numpy as np
import os

class PreProcessor:
    def __init__(self, path='', output_dir=r'./'):
        self.output_dir = output_dir
        if not path:
            self.dataLoader = DataLoader()
        else:
            self.dataLoader = DataLoader(path)
        self.classes_list, self.classes, self.dict_list = self.dataLoader.get_data()

    def process_data(self):
        full_dataset = np.empty_like(self.classes, dtype=list)

        for idx, dict in enumerate(self.dict_list):
            axes = []
            for row in dict:
                axes.append(float(row['gyroscope_x']))
                axes.append(float(row['gyroscope_y']))
                axes.append(float(row['gyroscope_z']))
                axes.append(float(row['acc_x']))
                axes.append(float(row['acc_y']))
                axes.append(float(row['acc_z']))
                
            if full_dataset[self.classes.index(self.classes_list[idx])]:
                full_dataset[self.classes.index(self.classes_list[idx])] = \
                    [*full_dataset[self.classes.index(self.classes_list[idx])], *axes]
            else:
                full_dataset[self.classes.index(self.classes_list[idx])] = [*axes]

        with h5py.File(self.output_dir + f'.{os.sep}data_custom_ml{os.sep}dataset.hdf5', 'w') as hf:
            for i in range(len(self.classes)):
                hf.create_dataset(self.classes[i], data=full_dataset[i])