import pdb

import torch
from torch.utils.data import Dataset
import json
import numpy as np

class Dataset(Dataset):
    def __init__(self, dataName, split, cMean=None, cStd=None, sMean=None, sStd=None):

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        with open('../DATA/{}_CURVATURE_{}.json'.format(self.split, dataName)) as j:
            self.curvature = json.load(j)

            # Loader를 쓰면 input의 shape가 다 똑같아야 하는건가?
            minlen = min([len(x) for x in self.curvature])
            self.curvature = np.array([x[:minlen] for x in self.curvature])

        with open('../DATA/{}_SPEED_{}.json'.format(self.split, dataName)) as j:
            self.speed = json.load(j)

        with open('../DATA/{}_SPEEDHIS_{}.json'.format(self.split, dataName)) as j:
            self.speedhist = json.load(j)

        self.dataset_size = len(self.curvature)

        self.cMean = cMean  # curvature
        self.cStd = cStd

        self.sMean = sMean  # speed
        self.sStd = sStd

        if self.cMean is None:
            curvature = np.array(self.curvature)
            self.cMean = np.mean(curvature)
            self.cStd = np.std(curvature)

            speed = np.array(self.speed)
            self.sMean = np.mean(speed)
            self.sStd = np.std(speed)

    def __getitem__(self, i):
        curvature = torch.FloatTensor(self.curvature[i])
        speed = torch.FloatTensor(self.speed[i])
        speedhist = torch.FloatTensor(self.speedhist[i])

        curvature = (curvature - self.cMean) / self.cStd
        speed = (speed - self.sMean) / self.sStd
        speedhist = (speedhist - self.sMean) / self.sStd

        return curvature, speed, speedhist

    def __len__(self):
        return self.dataset_size
