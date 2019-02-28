from torch.utils import data
import pandas as pd
import os


class TextDataset(data.Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, path):
        self.file_name = os.listdir(path)
        print(path+self.file_name[3])

        self.train_set = pd.read_csv(path+self.file_name[3]).values

        print(self.train_set.shape)
        print(self.train_set[0, 0], self.train_set[0, 2])
        # print(self.file_name)

    def __getitem__(self, index):
        return self.train_set[index, 2], self.train_set[index, 0]
        # return {'item': self.train_set[index, 2], 'label': self.train_set[index, 0]}

    def __len__(self):
        return len(self.train_set)