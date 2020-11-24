import pdb

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from random import random
from utils import *  # station_curvature

def calLateralOffset(local_xys, center_xys):
    '''함수 추가: center line에서 lateral offset 계산 (진행 방향의 오른쪽:+, 왼쪽:-) '''
    offsets = []
    for i, local_xy in enumerate(local_xys):
        eDist = np.array([np.sqrt(sum((xy - local_xy)**2)) for xy in center_xys])
        cur_idx_candi = np.where(eDist == eDist.min())
        cur_idx = cur_idx_candi[0][0]

        try:
            nn_center = center_xys[cur_idx]
            nn1_center = center_xys[cur_idx+1]
        except:
            nn_center = center_xys[cur_idx]
            nn1_center = center_xys[0]

        # move to Origin
        nn1_center_0 = np.array([nn1_center[0] - nn_center[0], nn1_center[1] - nn_center[1]])
        local_xy_0 = np.array([local_xy[0] - nn_center[0], local_xy[1] - nn_center[1]])

        # theta of heading of the center line
        theta = np.arccos(np.inner(nn1_center_0, np.array([1, 0]))/(np.linalg.norm(nn1_center_0)*np.linalg.norm(np.array([1, 0]))))
        if nn1_center_0[1] < 0:
            theta *= (-1)

        # transform -(theta)
        nn1_center_0_tr = np.inner(np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]), nn1_center_0)
        local_xy_0_tr = np.inner(np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]), local_xy_0)

        if local_xy_0_tr[1] >= 0:
            offsets.append(abs(local_xy_0_tr[1])*(-1))  # center line 왼쪽
        else:
            offsets.append(abs(local_xy_0_tr[1]))  # center line 오른쪽

        ## visualization
        # fig = plt.figure(figsize=(9, 3))
        # ax1 = fig.add_subplot(131)  # original
        # ax2 = fig.add_subplot(132)
        # ax3 = fig.add_subplot(133)
        # ax1.set_aspect('equal')
        # ax2.set_aspect('equal')
        # ax3.set_aspect('equal')

        # ax1.plot([nn_center[0], nn1_center[0]], [nn_center[1], nn1_center[1]], 'rx--')
        # ax1.plot([nn_center[0], local_xy[0]], [nn_center[1], local_xy[1]], 'bx--')

        # ax2.plot([0, nn1_center_0[0]], [0, nn1_center_0[1]], 'rx--')
        # ax2.plot([0, local_xy_0[0]], [0, local_xy_0[1]], 'bx--')
        # ax2.set_title(str(180/np.pi*theta))

        # ax3.plot([0, nn1_center_0_tr[0]], [0, nn1_center_0_tr[1]], 'rx--')
        # ax3.plot([0, local_xy_0_tr[0]], [0, local_xy_0_tr[1]], 'bx--')
        # ax3.set_title(str(offsets[-1]))

        # plt.suptitle(i)
        # plt.grid()
        # plt.show()

    print(local_xys.shape, offsets.__len__())
    return offsets

class Dataset(Dataset):
    def __init__(self, driver, circuits, curvatureLength, historyLength, predLength, cMean=None, cStd=None, vMean=None, vStd=None, aMean=None, aStd=None, lMean=None, lStd=None, vis=False):

        self.vis = vis

        driverDataFolder = '../Data/driverData/{}'.format(driver)
        mapDataFolder = '../Data/mapData'

        self.curvatureLength = curvatureLength
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

        for df, df_map, circuit in zip(self.dfs, self.df_maps, circuits): #''' 추가 '''
            if 'lateralOffset' in df.columns:
                continue
            else:
                print('Calculating LateralOffset ... (circuit: {})'.format(circuit))
                df['lateralOffset'] = calLateralOffset(df[['PosLocalX', 'PosLocalY']].values, df_map[['center_x', 'center_y']].values)
                df.to_csv(driverDataFolder + '/{}-record-scz_msgs.csv'.format(circuit))
                print('File Updated,', '/{}-record-scz_msgs.csv'.format(circuit))

        self.dataset_size = sum([len(df)-self.historyLength-self.predLength-1 for df in self.dfs])
        # accelFromV, a_t = v_(t+1) - v_t 이므로, 마지막 1개는 못 씀.

        self.cMean = cMean  # curvature
        self.cStd = cStd

        self.vMean = vMean  # speed
        self.vStd = vStd

        self.aMean = aMean  # accel
        self.aStd = aStd

        self.lMean = lMean  # lateral offset
        self.lStd = lStd

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

            lateralOffset = [] #''' 추가 '''
            for df in self.dfs:
                lateralOffset += df['lateralOffset'].values.tolist()
            lateralOffset = np.array(lateralOffset)
            self.lMean = np.mean(lateralOffset)
            self.lStd = np.std(lateralOffset)

    def __getitem__(self, i):
        dfIdx, idx = self.idx[i]

        df = self.dfs[dfIdx]
        speed = df['GPS_Speed'].values
        lateralOffset = df['lateralOffset'].values

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

        targetSpeeds = speed[idx:idx+self.predLength+1] # [0]: current Speed
        histSpeeds = speed[idx-self.historyLength:idx]
        targetAccels = np.diff(speed[idx:idx+self.predLength+2])  # [0]: a_t 부터 target이 되어야 함. vt에서 at를 예측해야 v_(t+1)로 넘어가니까.
        histAccels = np.diff(speed[idx-self.historyLength:idx+1])  # AttentionModel6에서 histAccel은 일단 안쓸거임.
        targetOffsets = lateralOffset[idx:idx+self.predLength+1] #''' 추가 '''
        histOffsets = lateralOffset[idx-self.historyLength:idx]

        curvature = torch.FloatTensor( (curvature - self.cMean) / self.cStd )
        targetSpeeds = torch.FloatTensor( (targetSpeeds - self.vMean) / self.vStd )
        histSpeeds = torch.FloatTensor( (histSpeeds - self.vMean) / self.vStd )
        targetOffsets = torch.FloatTensor( (targetOffsets - self.lMean) / self.lStd)
        histOffsets = torch.FloatTensor( (histOffsets - self.lMean) / self.lStd)

        if self.vis:
            return curvature, targetSpeeds, histSpeeds, targetAccels, histAccels, map_center_xy, curPosition, cur_idx_candi

        return curvature, targetSpeeds, histSpeeds, targetAccels, histAccels, targetOffsets, histOffsets

    def __len__(self):
        return self.dataset_size
