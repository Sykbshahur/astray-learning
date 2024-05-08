import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class ImageDataset_Test(Dataset):
    # def __init__(self, csv_file, img_size, filter_size, test_set):
    def __init__(self, csv_file, owntransforms):
        self.transform = owntransforms
        self.img_list = pd.read_csv(csv_file)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        path = self.img_list.iloc[index,0]
        if 'crop' in path:
            path = path[1:]
        else:
            path = path[1:8] + 'crop_img/' + path[8:]
        img = Image.open(path)
        label = np.array(self.img_list.iloc[index,1])
        img = self.transform(img)
        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.img_list)
