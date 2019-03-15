from torch.utils import data
import pandas as pd
import os


class TextDataset(data.Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    def __init__(self, path):

        self.file_name = os.listdir(path)
        self.train_set = pd.read_csv(path+self.file_name[3]).values
        # train_set is a 560000 x 3 matrix
        # the columns are: number_of_topic, heading_of_the_text, text
        # It is a text categorisation dataset:
        # there are 14 different topics and the texts are sentences

        print(path+self.file_name[3])
        print(self.train_set.shape)
        # print(self.train_set[0, 0], self.train_set[0, 2])
        # print(self.file_name)

    def __getitem__(self, index):
        # print(self.train_set[index, 2])   # this is the piece of text
        # print(self.train_set[index, 0])   # this is the label of the topic of the piece of text
        return self.train_set[index, 2], self.train_set[index, 0]

    def __len__(self):
        return len(self.train_set)