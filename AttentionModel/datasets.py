import pdb

import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np


class SpeedDataset(Dataset):
    '''
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    '''

    def __init__(self, data_folder, data_name, split, transform=None, mean=None, std=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load target speeds (completely into memory)
        with open(os.path.join(data_folder, self.split + '_SPEEDS_' + data_name + '.json'), 'r') as j:
            self.speeds = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.speeds)

        self.mean = mean
        self.std = std

        if self.mean is None:
            speeds = np.array(self.speeds)
            self.mean = np.mean(speeds)
            self.std = np.std(speeds)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        speed = torch.FloatTensor(self.speeds[i])

        speed = (speed - self.mean) / self.std  # normalization

        return img, speed

    def __len__(self):
        return self.dataset_size
