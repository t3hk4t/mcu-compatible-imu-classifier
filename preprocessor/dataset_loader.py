import csv
import matplotlib.pyplot as plt
import os
import json


class DataLoader:
    def __init__(self, data_location=f'data_raw'):
        self.data_dir = data_location
        self.config_dir = data_location + os.sep + "config.json"

        with open(self.config_dir) as fp:
            self.config = json.load(fp)
            self.classes = self.config['classes']

        self.classes_list = []
        self.dict_list = []
        self.__parse()

    def __parse(self):
        for filename in os.listdir(self.data_dir):
            if filename == 'config.json':
                continue
            f = os.path.join(self.data_dir, filename)

            for value in self.classes:
                if filename.startswith(value):
                    with open(f) as file:
                        reader = csv.DictReader(file)
                        self.dict_list.append(list(reader))
                        self.classes_list.append(value)
                        break
            

    def get_data(self):
        return self.classes_list, self.classes, self.dict_list # returns labeled classes, available classes and all the data
