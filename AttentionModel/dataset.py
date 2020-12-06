import pdb

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import random
from utils import *  # station_curvature

class Dataset(Dataset):
    def __init__(self, driver, circuits, curvatureLength, historyLength, predLength, cMean=None, cStd=None, vMean=None, vStd=None, aMean=None, aStd=None, vis=False):

        self.vis = vis

        driverDataFolder = '../Data/driverData/{}'.format(driver)
        mapDataFolder = '../Data/mapData'

        self.curvatureLength = curvatureLength  # parameters
        self.historyLength = historyLength
        self.predLength = predLength  # 0.1s * 20 = 2s

        self.numCircuits = len(circuits)

        self.dfs = [pd.read_csv(driverDataFolder + '/{}-record-scz_msgs.csv'.format(circuit)).dropna(how='all') for circuit in circuits]
        self.df_maps = [pd.read_csv(mapDataFolder + '/{}_norm.csv'.format(circuit)) for circuit in circuits]

        for df_map in self.df_maps:
            assert len(df_map) == round(df_map['distance'].values[-1])+1, 'Please normalize map data to have 1M unit'
            s, k = station_curvature(df_map['center_x'], df_map['center_y'])
            df_map['curvature'] = k

            # plt.figure()
            # plt.plot(df_map['center_x'], df_map['center_y'])
            # plt.show()

        self.idx = []
        for i in range(self.numCircuits):
            for j in range(self.historyLength, len(self.dfs[i])-self.predLength-1):
                self.idx.append([i, j])

        self.dataset_size = sum([len(df)-self.historyLength-self.predLength-1 for df in self.dfs])

        self.cMean = cMean  # curvature
        self.cStd = cStd

        self.vMean = vMean  # speed
        self.vStd = vStd

        self.aMean = aMean  # accel
        self.aStd = aStd

        if self.cMean is None:
            curvature = []
            for df_map in self.df_maps:
                curvature += df_map['curvature'].values.tolist()
            curvature = np.array(curvature)
            self.cMean = np.mean(curvature)
            self.cStd = np.std(curvature)

            speed = []
            for df in self.dfs:
                speed += df['GPS_Speed'].values.tolist()
            speed = np.array(speed)
            self.vMean = np.mean(speed)
            self.vStd = np.std(speed)

            self.aMean = np.mean(np.diff(speed))
            self.aStd = np.std(np.diff(speed))

    def __getitem__(self, i):
        dfIdx, idx = self.idx[i]

        df = self.dfs[dfIdx]
        speed = df['GPS_Speed'].values

        df_map = self.df_maps[dfIdx]
        map_center_xy = df_map[['center_x', 'center_y']].values
        curvature = df_map['curvature'].values

        curPosition = np.array([df.iloc[idx]['PosLocalX'], df.iloc[idx]['PosLocalY']])
        curPositionE = np.repeat(np.expand_dims(curPosition, axis=0), map_center_xy.shape[0], axis=0)
        eDist = np.sqrt(np.sum((curPositionE - map_center_xy)**2, 1))
        cur_idx_candi = np.where(eDist == eDist.min())[0][0]

        if cur_idx_candi + self.curvatureLength < len(df_map):
            curvature = curvature[cur_idx_candi+1:cur_idx_candi+self.curvatureLength+1]  # [0]: 1m ahead
        else:
            curvature = np.concatenate((curvature[cur_idx_candi+1:], curvature[:self.curvatureLength-(len(df_map)-cur_idx_candi-1)]))

        targetSpeeds = speed[idx:idx+self.predLength+1] # [0]: current Speed (v_(t))
        histSpeeds = speed[idx-self.historyLength:idx]
        targetAccels = np.diff(speed[idx:idx+self.predLength+2])  # [0]: a_(t) (v_(t+1) = v_(t) + a_(t))
        histAccels = np.diff(speed[idx-self.historyLength:idx+1])  # Not used.

        curvature = torch.FloatTensor( (curvature - self.cMean) / self.cStd )
        targetSpeeds = torch.FloatTensor( (targetSpeeds - self.vMean) / self.vStd )
        histSpeeds = torch.FloatTensor( (histSpeeds - self.vMean) / self.vStd )
        targetAccels = torch.FloatTensor( (targetAccels - self.aMean) / self.aStd )
        histAccels = torch.FloatTensor( (histAccels - self.aMean) / self.aStd )

        # curvature = curvature if random() > 0.5 else (-1)*curvature

        if self.vis:
            return curvature, targetSpeeds, histSpeeds, targetAccels, histAccels, map_center_xy, curPosition, cur_idx_candi

        return curvature, targetSpeeds, histSpeeds, targetAccels, histAccels

    def __len__(self):
        return self.dataset_size
