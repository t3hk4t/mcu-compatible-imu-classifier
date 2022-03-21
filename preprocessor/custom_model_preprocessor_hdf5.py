from parse_imu_log import DataLoader
import h5py
action_buffer = ['drive_off', 'drive_on', 'stand_off', 'stand_on', 'validate']


class PreProcessor:
    def __init__(self, overlap, path='', output_dir=r'./'):
        self.overlap = overlap
        self.output_dir = output_dir
        if not path:
            self.dataLoader = DataLoader()
        else:
            self.dataLoader = DataLoader(path)
        self.data = self.dataLoader.get_data()

    def process_data(self):
        full_dataset = []
        for dict in self.data:
            axes = []
            for row in dict:
                axes.append(float(row['gyroscope_x']))
                axes.append(float(row['gyroscope_y']))
                axes.append(float(row['gyroscope_z']))
                axes.append(float(row['acc_x']))
                axes.append(float(row['acc_y']))
                axes.append(float(row['acc_z']))
            full_dataset.append(list(axes))

        with h5py.File(self.output_dir + 'test_dataset.hdf5', 'w') as hf:
            for i in range(len(action_buffer)):
                hf.create_dataset(action_buffer[i], data=full_dataset[i])


test = PreProcessor(0)
test.process_data()

# read HDF5 file
with h5py.File('./' + 'test_dataset.hdf5', 'r') as hf:
    dset_x_train = hf['drive_off']
    dset_y_train = hf['drive_on']
    dset_x_test = hf['stand_off']
    dset_y_test = hf['stand_on']
    dset_y_validate = hf['validate']

    print(dset_x_train)
