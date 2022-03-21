import csv
import matplotlib.pyplot as plt
import os


class DataLoader:
    def __init__(self, data_location=r'..\data_raw'):
        self.data_dir = data_location
        self.dict_list = []
        self.__parse()

    def __parse(self):
        for filename in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, filename)
            if os.path.isfile(f):
                print("working on file: ", f)
            with open(f) as file:
                reader = csv.DictReader(file)
                self.dict_list.append(list(reader))

    def get_data(self):
        return self.dict_list

    def plot(self):
        for dict in self.dict_list:
            timestamp = []
            gyroscope_x = []
            gyroscope_y = []
            gyroscope_z = []
            acc_x = []
            acc_y = []
            acc_z = []

            for row in dict:
                timestamp.append(float(row['timestamp']))
                gyroscope_x.append(float(row['gyroscope_x']))
                gyroscope_y.append(float(row['gyroscope_y']))
                gyroscope_z.append(float(row['gyroscope_z']))
                acc_x.append(float(row['acc_x']))
                acc_y.append(float(row['acc_y']))
                acc_z.append(float(row['acc_z']))

            fig, ((ax1, ax2)) = plt.subplots(2, 1)
            ax1.plot(gyroscope_x, label='gyro_x')
            ax1.plot(gyroscope_y, label='gyro_y')
            ax1.plot(gyroscope_z, label='gyro_z')
            ax1.set_title('Gyroscope')
            ax2.plot(acc_x, label='acc_x')
            ax2.plot(acc_y, label='acc_y')
            ax2.plot(acc_z, label='acc_z')
            ax2.set_title('Accelerometer')
            plt.legend()
            plt.draw()
        plt.show()


Data = DataLoader()
Data.plot()