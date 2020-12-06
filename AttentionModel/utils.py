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
import scipy
from scipy import signal
from constants import device


def station_curvature(localX, localY, medfilt=15):
    # Menger's curvature
    dx = np.diff(localX)
    dy = np.diff(localY)
    l1 = np.sqrt(dx**2 + dy**2)                   # 1-0 부터
    l2 = l1[1:]  #np.sqrt(dx[1:]**2 + dy[1:]**2)  # 2-1 부터

    k1 = dx[:-1]/l1[:-1] + dx[1:]/l2
    k2 = dy[1:]/l2 - dy[:-1]/l1[:-1]
    k3 = l1[:-1] + l2

    k4 = dy[:-1]/l1[:-1] + dy[1:]/l2
    k5 = dx[1:]/l2 - dx[:-1]/l1[:-1]

    k = ((k1 * k2) / k3) - ((k4 * k5) / k3)
    if dx[dx==0].__len__() > 0 or dy[dy==0].__len__() > 0:
        pdb.set_trace()
    k = np.insert(k, 0, k[0])
    k = np.insert(k, k.size, k[-1])

    k[k>0.5] = 0.5
    k[k<-0.5] = -0.5
    k = signal.medfilt(k, medfilt)

    return np.cumsum(l1), k

def transform_plane(localX, localY, heading):
    m = np.array([[np.sin(heading), np.cos(heading)], [np.cos(heading), -np.sin(heading)]])
    preview = m.dot(np.vstack(([localX],[localY])))
    return preview[0], preview[1]

def replicateMapInfo(mapFile, df, vis=False, startX=None, startY=None):
    ## replicate df_map: 1) counting the number of laps, 2) replicate map info
    try:
        df_map = pd.read_csv(mapFile)
    except:
        df_map = mapFile  # 이미 df 인 경우

    # partitioning each lap
    if startX is None:
        startX = df['PosLocalX'][0]
        startY = df['PosLocalY'][0]
    # gap = 10  # location threshold, user-defined  # for YJ
    # gap2 = 800  # new lap threshold, user-defined
    # gap = 20  # location threshold, user-defined  # for YS
    # gap = 30  # for YS, mugello
    gap  = 20  # for YS, ref with yj, spa
    # gap = 350 # for YS, ref with yj, imola
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
    df_map = pd.concat([df_map]*(len(startingPoints)+1), ignore_index=True)  # curvature를 위해 +1

    dist = df_map_dist.tolist()
    last_dist_v = df_map_dist[-1]
    for i in range(len(startingPoints)):
        # print(i, dist[-1], print(df_map_dist + (last_dist_v+1)))
        dist += (df_map_dist + (last_dist_v+1)).tolist()
        last_dist_v = dist[-1]
    dist = np.array(dist)
    df_map['distance'] = dist

    try:
        df_map_x = (df_map['inner_x'].values + df_map['outer_x'].values)/2
        df_map_y = (df_map['inner_y'].values + df_map['outer_y'].values)/2
        map_center_xy = np.column_stack((df_map_x, df_map_y))
    except:
        map_center_xy = np.column_stack((df_map['center_x'], df_map['center_y']))

    return df_map, map_center_xy

# def getMapPreview(i, df, df_map, map_center_xy, previewDistance, dh, transform=False, headingInfo=None):
def getMapPreview(i, df, df_map, map_center_xy, previewDistance, transform=False, headingInfo=None):
    cur_xy = np.array([df.iloc[i]['PosLocalX'], df.iloc[i]['PosLocalY']])
    eDist = np.array([np.sqrt(sum((xy - cur_xy)**2)) for xy in map_center_xy])

    try:
        # cur_idx = np.argmin(eDist)  # local x, y도 반복되기 때문에, 다시 i에서 가까운 것으로 해야 함.
        cur_idx_candi = np.where(eDist == eDist.min())
        temp = (cur_idx_candi[0] - i)  # i를 뺀 값임. 왜냐하면, 맵을 빙글빙글 도니까, i 다음 번을 가져야 할거 같아서
        # cur_idx = np.min(temp[temp>=0])  # i를 뺀 값을 그대로 index로 쓰면 엉뚱한 곳을 preview로 가져오게 됨.
        cur_idx = cur_idx_candi[0][np.where(temp>0)[0][0]]
        cur_dist = df_map['distance'][cur_idx]  # map의 시작을 0으로 했을 때, cur_xy까지의 거리
        preview_end_idx = np.argmin(abs(df_map['distance'].values - (cur_dist + previewDistance)))
    except:
        return [], [], [], []

    map_preview = map_center_xy[cur_idx:preview_end_idx+1, :]  # 그림 그릴때 쓰는 함수니까. curvature[0]을 현재 위치로.
    dist, curvature = station_curvature(map_preview[:, 0], map_preview[:, 1], medfilt=15)

    if transform:
        transX, transY = transform_plane(map_preview[:, 0]-map_preview[0, 0],
                                            map_preview[:, 1]-map_preview[0, 1], headingInfo[cur_idx])
        return dist, curvature, map_preview, np.column_stack((transX, transY))

    else:
        return dist, curvature, map_preview

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

def pearsonr(predict, true):
    predict = predict.cpu().detach().numpy()
    true = true.cpu().detach().numpy()

    batch_size = predict.shape[0]
    vCorr = 0
    vnumNaN = 0  # vCorr도 NaN이 생기는 경우가 있음. validation으로 magione을 썼을 때 나타남. true가 0벡터인 경우
    for i in range(batch_size):
        corr = scipy.stats.pearsonr(predict[i, :, 0], true[i, :, 0])[0]
        if np.isnan(corr):
            vnumNaN += 1
        else:
            vCorr += corr

    if predict.shape[2] > 1:
        aCorr = 0
        numNaN = 0
        for i in range(batch_size):
            corr = scipy.stats.pearsonr(predict[i, :, 1], true[i, :, 1])[0]
            if np.isnan(corr):
                numNaN += 1
            else:
                aCorr += corr

        if vnumNaN == batch_size:
            if numNaN == batch_size:
                return np.nan, np.nan
            else:
                return np.nan, aCorr/(batch_size - numNaN)
        else:
            if numNaN == batch_size:
                return vCorr/(batch_size - vnumNaN), np.nan

        return vCorr/(batch_size - vnumNaN), aCorr/(batch_size - numNaN)

    if batch_size == vnumNaN:
        return np.nan
    return vCorr/(batch_size - vnumNaN)

def save_checkpoint(dataFolderPath, encoder, decoder, epoch, cMean_tr, cStd_tr, vMean_tr, vStd_tr, aMean_tr, aStd_tr, curvatureLength, historyLength):
                    # loss_tr_list, loss_vl_list, mape_tr_list, mape_vl_list, rmse_tr_list, rmse_vl_list, RMSE=False):
    state = {'epoch': epoch,
             'cMean_tr': cMean_tr,
             'cStd_tr': cStd_tr,
             'vMean_tr': vMean_tr,
             'vStd_tr': vStd_tr,
             'aMean_tr': aMean_tr,
             'aStd_tr': aStd_tr,
             'curvatureLength': curvatureLength,
             'historyLength': historyLength
             # 'loss_tr_list': loss_tr_list,
             # 'loss_vl_list': loss_vl_list,
             # 'mape_tr_list': mape_tr_list,
             # 'mape_vl_list': mape_vl_list,
             # 'rmse_tr_list': rmse_tr_list,
             # 'rmse_vl_list': rmse_vl_list
             }

    enc_filename = '{}/checkpoint_ENC_{}.pth.tar'.format(dataFolderPath, epoch)
    dec_filename = '{}/checkpoint_DEC_{}.pth.tar'.format(dataFolderPath, epoch)
    stat_filename = '{}/stat_{}.pickle'.format(dataFolderPath, epoch)

    torch.save(encoder.state_dict(), enc_filename)
    torch.save(decoder.state_dict(), dec_filename)
    with open(stat_filename, 'wb') as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

def adjust_learning_rate(epoch, optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nEpoch {}: DECAYING learning rate.".format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def reset_learning_rate(epoch, optimizer, value):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    """

    print("\nEpoch {}: RESET learning rate --> {}.".format(epoch, value))
    for param_group in optimizer.param_groups:
        param_group['lr'] = value

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def weightedMseNslopeLoss(input, target):  # weighted MSE
    weight = (torch.linspace(100, 70, input.size(1)))
    weight = weight.unsqueeze(0).unsqueeze(2).to(device)
    # slopeMSE = torch.mean(((input[:, 1:, 0] - input[:, :-1, 0]) - (target[:, 1:, 0] - target[:, :-1, 0]))**2)
    MSE = torch.mean(weight*(input-target)**2)
    return MSE

def MSEwSmoothing(input, target, mean, std):
    MSE = torch.mean((input-target)**2)
    # wMSE = torch.mean(weight*(input-target)**2)
    # vMSE = torch.mean((input[:, :, 0]-target[:, :, 0])**2)
    # aMSE = torch.mean((input[:, :, 1]-target[:, :, 1])**2)

    # smoothing = torch.mean(torch.abs(input[:, 1:] - input[:, :-1]))
    vsmoothing = torch.mean(torch.abs(input[:, 1:, 0] - input[:, :-1, 0]))
    # asmoothing = torch.mean(torch.abs(input[:, 1:, 1] - input[:, :-1, 1]))

    # return wMSE + 100*vsmoothing + 110*asmoothing
    # return wMSE + 75*vsmoothing + 75*asmoothing
    # return MSE + 0.8*vsmoothing + asmoothing
    return MSE + vsmoothing, MSE, vsmoothing

def MSEwSmoothingwConsistency(input, target, mean, std):
    MSE = torch.mean((input-target)**2)
    vMSE = torch.mean((input[:, :, 0] - target[:, :, 0])**2)
    aMSE = torch.mean((input[:, :, 1] - target[:, :, 1])**2)

    smoothing = torch.mean(torch.abs(input[:, 1:, :] - input[:, :-1, :]))

    fp = 5
    fsmoothing = torch.mean(torch.abs(input[:, 1:fp+1, 0] - input[:, :fp, 0]))

    vsmoothing = torch.mean(torch.abs(input[:, 1:, 0] - input[:, :-1, 0]))
    asmoothing = torch.mean(torch.abs(input[:, 1:, 1] - input[:, :-1, 1]))

    input_unnorm = input*std+mean
    pred_v = input_unnorm[:, :, 0]
    pred_a = input_unnorm[:, :, 1]

    diff_pred_v = pred_v[:, 1:] - pred_v[:, :-1]

    consistencyVA = torch.mean(torch.abs(diff_pred_v - pred_a[:, :-1]))

    return MSE + 0.01*consistencyVA, (MSE, vMSE, aMSE, consistencyVA)

def aMSEwConsistency(input, target, mean, std):
    MSE = torch.mean((input-target)**2)
    vMSE = torch.mean((input[:, :, 0] - target[:, :, 0])**2)
    aMSE = torch.mean((input[:, :, 1] - target[:, :, 1])**2)

    input_unnorm = input*std+mean
    pred_v = input_unnorm[:, :, 0]
    pred_a = input_unnorm[:, :, 1]

    diff_pred_v = pred_v[:, 1:] - pred_v[:, :-1]

    consistencyVA = torch.mean(torch.abs(diff_pred_v - pred_a[:, :-1]))

    return aMSE + 0.001*consistencyVA, (MSE, vMSE, aMSE, consistencyVA)

def vaMSEwConsistency(input, target, mean, std):
    MSE = torch.mean((input-target)**2)
    vMSE = torch.mean((input[:, :, 0] - target[:, :, 0])**2)
    aMSE = torch.mean((input[:, :, 1] - target[:, :, 1])**2)

    input_unnorm = input*std+mean
    pred_v = input_unnorm[:, :, 0]
    pred_a = input_unnorm[:, :, 1]

    diff_pred_v = pred_v[:, 1:] - pred_v[:, :-1]

    consistencyVA = torch.mean(torch.abs(diff_pred_v - pred_a[:, :-1]))

    return 0.1*vMSE + aMSE + 0.01*consistencyVA, (MSE, vMSE, aMSE, consistencyVA)

def MSE(input, target, mean, std):
    MSE = torch.mean((input-target)**2)
    vMSE = 0
    aMSE = 0
    consistencyVA = 0

    return MSE, (MSE, vMSE, aMSE, consistencyVA)

def MSEwAtt(input, target, mean, std, alphas, alphas_target):
    MSE = torch.mean((input-target)**2)
    vMSE = 0  # Not tracked
    aMSE = 0  # Not tracked
    consistencyVA = 0  # Not tracked
    ATT = abs(torch.mean((alphas - alphas_target)))

    return MSE+100*ATT, (MSE, vMSE, aMSE, consistencyVA, 100*ATT)

def L1(input, target, mean, std):
    MSE = torch.sum((input-target)**2)
    vMSE = 0
    aMSE = 0
    consistencyVA = 0

    return MSE, (MSE, vMSE, aMSE, consistencyVA)

def smooth(input, weight):
    last = input[0]
    smoothed = []
    for point in input:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

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

if __name__ == "__main__":
    driver='YJ'
    circuit='YYF'
    mapFile = '../Data/mapData/{}-norm.csv'.format(circuit)
    recordFile = '../Data/driverData/{}/{}-record-scz_msgs.csv'.format(driver, circuit)
    df = pd.read_csv(recordFile)
    replicateMapInfo(mapFile, df, vis=True)

    # ref_recordFile = '../Data/driverData/{}/{}-record-scz_msgs.csv'.format('YJ', circuit)
    # ref_df = pd.read_csv(ref_recordFile)
    # replicateMapInfo(mapFile, df, vis=True, startX=ref_df['PosLocalX'][0], startY=ref_df['PosLocalY'][0])
