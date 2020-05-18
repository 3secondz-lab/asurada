
import pdb

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class dataNormalization:
    def __init__(self, data):
        self.mu, self.std = norm.fit(data)
        self.data = (data-self.mu)/self.std

    def normalization(self, data):
        return (data-self.mu)/self.std

    def denormalization(self, data):
        return (data*self.std) + self.mu

def files2df(DATA_PATH, files):
    # This function originally aimed to read multiple data files into one data set,
    # but currently only supports one data file. (시간 순서 때문에 현재 함수로는 여러개의 파일을 합칠 수가 없음.)
    df_list = []
    for file in files:
        filepath = os.path.join(DATA_PATH, file)
        assert os.path.exists(filepath), 'No filename {}'.format(file)
        df_list.append(pd.read_csv(filepath))
    return pd.concat(df_list)

def draw_result_graph_fitting(true, predict, curvature, previewType, predictLength, true_curv=None,
                                idxFrom=0, idxTo=None):
    if idxTo is None:
        idxTo = len(true)

    fig, ax = plt.subplots()
    line1 = ax.plot(true[idxFrom:idxTo], 'b.-', label='True', linewidth=2)
    line2 = ax.plot(predict[idxFrom:idxTo], 'r.-', label='Predict', linewidth=2)

    ax1 = ax.twinx()
    line3 = ax1.plot([(-1)*x for x in curvature[idxFrom:idxTo]], 'k--', label='Curvature', linewidth=1)

    if true_curv is not None:
        line4 = ax.plot(true_curv[idxFrom:idxTo], 'b--', label='True_curv', linewidth=1)
        lines = line1 + line2 + line3 + line4
    else:
        lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs)

    rmse = np.sqrt(sum((true[idxFrom:idxTo] - predict[idxFrom:idxTo])**2)/true[idxFrom:idxTo].shape[0])
    mape = 100*sum(abs(true[idxFrom:idxTo] - predict[idxFrom:idxTo])/true[idxFrom:idxTo])/true[idxFrom:idxTo].shape[0]
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print(' ')

    if previewType == 'TIME':
        ax.set_title('{}s ahead (RMSE:{:.3f}, MAPE:{:.3f})'.format(predictLength, rmse, mape))
    elif previewType == 'DISTANCE':
        pass
    ax.set_xlabel('Time [0.1s]')
    ax.set_ylabel('GPS_Speed')
    ax1.set_ylabel('(-1)*|Curvature|')
    plt.show()

def draw_result_graph(true, predict):
    pass

def buildDataset4fit(df, previewHelper, previewType):
    ks = []
    for idx in range(len(df)):
        print('Building training set... {}/{}'.format(idx, len(df)), end='\r')

        preview = previewHelper.get_preview(idx, previewType)
        ks.append(preview['Curvature'])

    pdb.set_trace()
    pad = len(max(ks, key=len))  # just for saving data pair as .npy
    ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])

    return ks_arr
