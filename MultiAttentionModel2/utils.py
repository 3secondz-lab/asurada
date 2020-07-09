import pdb

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper

import numpy as np
import pandas as pd
from helper import DataHelper
from tqdm import tqdm
from random import random
import json
import matplotlib.pyplot as plt

import torch
import pickle

'''
output_dataset = {data: [{'previewCurvature': np.ndarray, [0]: currentCurvature ~0
                        # 'previewCurvatureMean': float,
                        # 'currentSpeed': float,
                          'targetSpeeds': np.ndarray, [0]: currentSpeed
                          'split': str,
                          'historySpeed': np.ndarray [-1]: v_(t-1) }, ],
                  previewType: str,
                  previewTime: float [sec.],
                  previewDistance: float [m],
                  historyTime: float [sec.],
                  recFreq: int [Hz] }
'''

def replicateMapInfo(mapFile, df, vis=False):
    ## replicate df_map: 1) counting the number of laps, 2) replicate map info
    df_map = pd.read_csv(mapFile)

    # partitioning each lap
    startX = df['PosLocalX'][0]
    startY = df['PosLocalY'][0]
    gap = 10  # location threshold, user-defined
    gap2 = 800  # new lap threshold, user-defined
    startingPoints = df[(df['PosLocalX'] > (startX - gap)) & (df['PosLocalX'] < (startX + gap)) & (df['PosLocalY'] > (startY - gap)) & (df['PosLocalY'] < (startY + gap))].index
    startingPoints = [0] + [y for x, y in zip(startingPoints, startingPoints[1:]) if y-x > gap2] + [len(df)]

    if vis:
        print(startingPoints)
        plt.figure()
        plt.plot(df['GPS_Speed'])
        x = [*np.arange(0, 200, .1)]
        for point in startingPoints:
            plt.plot([point]*len(x), x, 'r--')
        plt.show()

    df_map_dist = df_map['distance'].values
    df_map = pd.concat([df_map]*len(startingPoints), ignore_index=True)

    dist = df_map_dist.tolist()
    last_dist_v = df_map_dist[-1]
    for i in range(len(startingPoints)-1):
        dist += (df_map_dist + last_dist_v).tolist()
        last_dist_v = dist[-1]
    dist = np.array(dist)

    df_map['distance'] = dist

    df_map_x = (df_map['inner_x'].values + df_map['outer_x'].values)/2
    df_map_y = (df_map['inner_y'].values + df_map['outer_y'].values)/2
    map_center_xy = np.column_stack((df_map_x, df_map_y))

    return df_map, map_center_xy

def getMapPreview(i, df, df_map, map_center_xy, previewDistance, dh):
    cur_xy = np.array([df.iloc[i]['PosLocalX'], df.iloc[i]['PosLocalY']])
    eDist = np.array([np.sqrt(sum((xy - cur_xy)**2)) for xy in map_center_xy])

    cur_idx_candi = np.where(eDist == eDist.min())
    temp = (cur_idx_candi[0] - i)
    cur_idx = np.min(temp[temp>=0])
    # cur_idx = np.argmin(eDist)  # local x, y도 반복되기 때문에, 다시 i에서 가까운 것으로 해야 함.
    cur_dist = df_map['distance'][cur_idx]  # map의 시작을 0으로 했을 때, cur_xy까지의 거리
    preview_end_idx = np.argmin(abs(df_map['distance'].values - (cur_dist + previewDistance)))

    map_preview = map_center_xy[cur_idx:preview_end_idx+1, :]

    ss, kk = dh.station_curvature(map_preview[:, 0], map_preview[:, 1], medfilt=15)
    return ss, kk

def create_dataset(drdFiles_tr, drdFiles_vl, drdFiles_te, recFreq,
                   previewType=None, previewTime=None, previewDistance=None,
                   trRate=0.8, output_path=None, output_fname=None, historyTime=None, hasCircuitInfo=False):

    if hasCircuitInfo:
        drdList = [(x, y, 'Train') for x, y in drdFiles_tr] + [(x, y, 'Val') for x, y in drdFiles_vl] + [(x, y, 'Test') for x, y in drdFiles_te]
    else:
        drdList = [(x, None, 'Train') for x in drdFiles_tr] + [(x, None, 'Val') for x in drdFiles_vl] + [(x, None, 'Test') for x in drdFiles_te]

    dataset = []  # for output_dataset['data']
    for drdFile, mapFile, dataType in drdList:
        print('Read Data...', drdFile, mapFile, dataType)

        df = pd.read_csv(drdFile)  # drdFile
        if 'Speed' in df.columns and 'GPS_Speed' not in df.columns:
            df = df.rename(columns={'Speed':'GPS_Speed'})

        if hasCircuitInfo:
            assert mapFile is not None, 'Check whether you have mapFile'
            df_map, map_center_xy = replicateMapInfo(mapFile, df)

        # helper
        dh_time = DataHelper(df)  # for target variables
        dh_time.set_preview_time(previewTime)

        dh_dist = DataHelper(df)  # for calculating preview curvature without mapfile
        dh_dist.set_preview_distance(previewDistance)

        for i in tqdm(range(len(df))):
            data = {}

            ''' ===== Input Data fields ===== '''
            ''' Preview-related: previewCurvature, previewDistance, previewHeight '''
            if hasCircuitInfo:  # 현재 local position과 가장 가까운 map_xy로부터 previewDistance 만큼의 curvature를 계산

                ss, kk = getMapPreview(i, df, df_map, map_center_xy, previewDistance, dh_time)
                # # doesn't matter which one is used for dh (dh_time, dh_dist)

                data['previewDistance'] = ss.tolist()
                data['previewCurvature'] = kk.tolist()
                # data['previewHeight'] =   # map 정보에 없음...

            else:
                preview_dist = dh_dist.get_preview(i, method='DISTANCE')
                data['previewCurvature'] = preview_dist['Curvature'].tolist()  # np.ndarray -> list
                data['previewDistance'] = preview_dist['Distance'].tolist()

            ''' History-related: GPS_Speed, AngleSteer, PedalPosAcc, PedalPosBrk '''  # data가 시간 순으로 기록되어 있음.
            data['historySpeeds'] = df['GPS_Speed'].values[i-int(recFreq*historyTime):i].tolist()
            data['historyAngles'] = df['AngleSteer'].values[i-int(recFreq*historyTime):i].tolist()
            data['historyAcc'] = df['PedalPosAcc'].values[i-int(recFreq*historyTime):i].tolist()
            data['historyBrk'] = df['PedalPosBrk'].values[i-int(recFreq*historyTime):i].tolist()

            ''' ===== Output Data fields ===== '''
            ''' Time-related: GPS_Speed, AngleSteer, PedalPosAcc, PedalPosBrk '''
            preview_time = dh_time.get_preview(i, method='TIME')
            data['targetSpeeds'] = preview_time['GPS_Speed'].tolist()
            data['targetAngles'] = preview_time['AngleSteer'].tolist()
            data['targetAcc'] = preview_time['PedalPosAcc'].tolist()
            data['targetBrk'] = preview_time['PedalPosBrk'].tolist()


            if dataType == 'Train':
                # data['split'] = 'val' if random() > trRate else 'train'  # When there is no validation data
                data['split'] = 'train'
            elif dataType == 'Val':
                data['split'] = 'val'
            else:
                data['split'] = 'test'

            dataset.append(data)

    output_dataset = {}
    output_dataset['data'] = dataset
    # output_dataset['previewType'] = previewType
    output_dataset['previewTime'] = previewTime
    output_dataset['previewDistance'] = previewDistance
    output_dataset['historyTime'] = historyTime
    output_dataset['recFreq'] = recFreq

    with open('{}/{}.json'.format(output_path, output_fname), 'w') as j:
        json.dump(output_dataset, j)

def create_input_files(jsonPath, output_path, output_fname,
                       cWindow=20, vWindow=2, vpWindow=2, cUnit=10, vUnit=10, vpUnit=10):
    assert os.path.isfile(jsonPath), 'Run create_dataset() first'

    with open(jsonPath, 'r') as j:
        dataset = json.load(j)


    recFreq = dataset['recFreq']
    # assertion for cWindow, vWindow, cUnit, vUnit (?)

    data = dataset['data']

    tr_curvature = []
    val_curvature = []
    test_curvature = []

    tr_targetSpeed = []
    val_targetSpeed = []
    test_targetSpeed = []

    tr_historySpeed = []
    val_historySpeed = []
    test_historySpeed = []

    for d in data:
        if d['split'] == 'train':
            # tr_curvature.append(d['previewCurvature'])
            tr_curvature.append((d['previewCurvature'], d['previewDistance']))
            tr_targetSpeed.append(d['targetSpeeds'])
            tr_historySpeed.append(d['historySpeeds'])
        elif d['split'] == 'val':
            # val_curvature.append(d['previewCurvature'])
            val_curvature.append((d['previewCurvature'], d['previewDistance']))
            val_targetSpeed.append(d['targetSpeeds'])
            val_historySpeed.append(d['historySpeeds'])
        elif d['split'] == 'test':
            # test_curvature.append(d['previewCurvature'])
            test_curvature.append((d['previewCurvature'], d['previewDistance']))
            test_targetSpeed.append(d['targetSpeeds'])
            test_historySpeed.append(d['historySpeeds'])

    for cs_ds, vs, vps, split in [(tr_curvature, tr_targetSpeed, tr_historySpeed, 'TRAIN'),
                               (val_curvature, val_targetSpeed, val_historySpeed, 'VAL'),
                               (test_curvature, test_targetSpeed, test_historySpeed, 'TEST')]:
        curvatures = []
        speeds = []
        historySpeeds = []

        for i in tqdm(range(len(cs_ds))):
            # if len(cs[i]) < recFreq*cWindow+1:  # [0]: current curvature ~0
            #     continue
            cs = cs_ds[i][0]  # previewCurvature
            ds = cs_ds[i][1]  # previewDistance

            if abs(ds[-1]-cWindow) > 5:
                continue
            if len(vs[i]) < recFreq*vWindow+1:
                continue
            if len(vps[i]) < recFreq*vpWindow:
                continue

            # ctemp = cs[i][:recFreq*cWindow+1]
            # ctemp = [ctemp[i] for i in range(len(ctemp)) if i%int(recFreq/cUnit)==0]
            ctemp = cs[:]

            vtemp = vs[i][:recFreq*vWindow+1]
            vtemp = [vtemp[i] for i in range(len(vtemp)) if i%int(recFreq/vUnit)==0]

            vptemp = vps[i][-1*(recFreq*vpWindow):]  # (~:v_(t-1))
            vptemp = [vptemp[i] for i in range(len(vptemp)) if i%int(recFreq/vpUnit)==0]

            curvatures.append(ctemp)
            speeds.append(vtemp)
            historySpeeds.append(vptemp)

        assert len(curvatures) == len(speeds)

        with open('{}/{}_CURVATURE_{}_{}_{}_{}_{}_{}_{}.json'.format(output_path, split, output_fname, cWindow, vWindow, vpWindow, cUnit, vUnit, vpUnit), 'w') as j:
            json.dump(curvatures, j)
        with open('{}/{}_SPEED_{}_{}_{}_{}_{}_{}_{}.json'.format(output_path, split, output_fname, cWindow, vWindow, vpWindow, cUnit, vUnit, vpUnit), 'w') as j:
            json.dump(speeds, j)
        with open('{}/{}_SPEEDHIS_{}_{}_{}_{}_{}_{}_{}.json'.format(output_path, split, output_fname, cWindow, vWindow, vpWindow, cUnit, vUnit, vpUnit), 'w') as j:
            json.dump(historySpeeds, j)

def accuracy(predict, true, excludeZero=True):
    if excludeZero:
        zeroIdx = (true == 0).nonzero()[:, 0].unique()
        nonZeroIdx = [x for x in range(true.shape[0]) if x not in zeroIdx]
        try:
            mape = torch.mean(torch.abs(predict[nonZeroIdx, :]-true[nonZeroIdx, :])/true[nonZeroIdx, :]*100)
        except:
            mape = torch.mean(torch.abs(predict[nonZeroIdx]-true[nonZeroIdx])/true[nonZeroIdx]*100)
    else:
        mape = torch.mean(torch.abs(predict-true)/true*100)

    rmse = torch.sqrt(torch.mean((predict-true)**2))
    return mape.item(), rmse.item()

def save_checkpoint(dataName, epoch, encoder_c, encoder_d, decoder, cMean_tr, cStd_tr, sMean_tr, sStd_tr, is_best,
                    loss_tr_list, loss_vl_list, mape_tr_list, mape_vl_list, rmse_tr_list, rmse_vl_list, RMSE=False):
    state = {'epoch': epoch,
             'cMean_tr': cMean_tr,
             'cStd_tr': cStd_tr,
             'sMean_tr': sMean_tr,
             'sStd_tr': sStd_tr,
             'loss_tr_list': loss_tr_list,
             'loss_vl_list': loss_vl_list,
             'mape_tr_list': mape_tr_list,
             'mape_vl_list': mape_vl_list,
             'rmse_tr_list': rmse_tr_list,
             'rmse_vl_list': rmse_vl_list}

    encC_filename = './chpt_yj/checkpoint_ENCC_' + dataName + '_{}.pth.tar'.format(epoch)
    encD_filename = './chpt_yj/checkpoint_ENCD_' + dataName + '_{}.pth.tar'.format(epoch)
    dec_filename = './chpt_yj/checkpoint_DEC_' + dataName + '_{}.pth.tar'.format(epoch)
    stat_filename = './chpt_yj/stat_' + dataName + '_{}.pickle'.format(epoch)

    torch.save(encoder_c.state_dict(), encC_filename)
    torch.save(encoder_d.state_dict(), encD_filename)
    torch.save(decoder.state_dict(), dec_filename)
    with open(stat_filename, 'wb') as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    # if is_best:
    #     if not RMSE:  # mape
    #         torch.save(encoder_c.state_dict(), 'BEST_' + encC_filename)
    #         torch.save(encoder_d.state_dict(), 'BEST_' + encD_filename)
    #         torch.save(decoder.state_dict(), 'BEST_' + dec_filename)
    #         with open('BEST_' + stat_filename, 'wb') as f:
    #             pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
    #     else:  # rmse
    #         torch.save(encoder_c.state_dict(), 'BEST_RMSE_' + encC_filename)
    #         torch.save(encoder_d.state_dict(), 'BEST_RMSE_' + encD_filename)
    #         torch.save(decoder.state_dict(), 'BEST_RMSE_' + dec_filename)
    #         with open('BEST_RMSE_' + stat_filename, 'wb') as f:
    #             pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
