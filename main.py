
import pdb

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # plot에서만 (lon, lat)순서로, 나머지는 반대로가 convention

from helper import DataHelper

import Model
from Utils import *
from GraphUtils import *
from network import *

try1 = True  # polyfit
try2 = False  # 고치는 중 planefit

''' Path Setting (Data, Model) '''  # 데이터는 데이터 폴더로, 모델은 모델 폴더로 정리 해야 할듯.
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(current_dir, 'DATA')


''' Training/Test Data Setting
    * You can use multiple record files to make one train/test data.
    * You can use the functions of GraphUtils to visualize data.
        - AnimatedPlots
'''
datafiles_tr = ['std_001.csv']
df_tr = files2df(DATA_PATH, datafiles_tr)

datafiles_te = ['std_002.csv']
df_te = files2df(DATA_PATH, datafiles_te)


''' Preview Helper for train/test '''
previewHelper_tr = DataHelper(df_tr)
previewHelper_te = DataHelper(df_te)  # preview needed when the test phase

previewType = 'TIME'
# previewType = 'DISTANCE'

preview_time = 5  # unit: [s]
previewHelper_tr.set_preview_time(preview_time)
previewHelper_te.set_preview_time(preview_time)

preview_distance = 250  # unit: [m]
previewHelper_tr.set_preview_distance(preview_distance)
previewHelper_te.set_preview_distance(preview_distance)


''' ######################################## '''
''' Speed prediction using preview curvature '''
''' ######################################## '''

''' [Try1]: Using GPS_Speed: v(t+1) = f(k_(t+1)) '''
if try1:
    predict_length = 30  # predict the speed at 1s ahead
    # carSpeed = df_te['GPS_Speed'].values
    agent = Agent(df_tr, 'tr-01-01', previewHelper_tr, previewType,\
                                    100, 100, 64, 30, 0.2, 64, \
                    use_throttle=True, use_steer_spd=True)
    # training
    input, output = agent.preprocess(df_tr)
    agent.update(input, output)
    # load
    # agent.load(30000)
    # predicts, true_vals, abs_curvatures, len_predicts = agent.test(df_te)


    pdb.set_trace()
    # Graph
    for i in range(predicts.shape[1]):
    # for i in range(10):
        fig, ax = plt.subplots()
        line1 = ax.plot(true_vals[:, i], label='True', linewidth=2)
        line2 = ax.plot(predicts[:, i], label='Predict', linewidth=2)

        ax1 = ax.twinx()
        line3 = ax1.plot([(-1)*x[i] for x in abs_curvatures], 'k--', label='Curvature', linewidth=1)

        lines = line1 + line2 + line3
        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs)

        rmse = np.sqrt(sum((true_vals[:, i] - predicts[:, i])**2)/true_vals.shape[0])
        mape = 100*sum(abs(true_vals[:, i] - predicts[:, i])/true_vals[:, i])/true_vals.shape[0]

        ax.set_title('{}s ahead (RMSE:{:.3f}, MAPE:{:.3f})'.format((i+1)/10, rmse, mape))
        ax.set_xlabel('Time [0.1s]')
        ax.set_ylabel('GPS_Speed')
        ax1.set_ylabel('(-1)*|Curvature|')
    plt.show()


''' [Try2]: Using Diff(GPS_Speed): v(t+1) = v(t) + f(k_(t+1:t+k), v_(t)) '''
if try2:
    # PolyFit = Model.PolyFit(df_tr, 'tr-01-01', previewHelper_tr, previewType,
    #                                 order=7, xlabel='abs(k)', ylabel='(-1)*v', vis=0)
    #
    # predict_length = 10  # predict the speed at 1s ahead
    # carSpeed = df_te['GPS_Speed'].values
    #
    # predict, abs_curvature, last_valid_idx = PolyFit.test(df_te, previewHelper_te, previewType, predict_length)
    # true = np.array([carSpeed[idx:idx+predict_length] for idx in range(last_valid_idx+1)])

    ##################
    predict_length = 10  # 1s
    PlaneFit = Model.PlaneFit(df_tr, 'tr-01-01', previewHelper_tr, previewType, predict_length,
        order=2, xlabel='(-1)*speed', ylabel='feature of abs(k)', zlabel='speedDiff', vis=1)

    ###################### 밥 먹고 여기서부터 고치자.... ㅜ.ㅜ
    carSpeed = df_te['GPS_Speed'].values
    predict, last_valid_idx = PlaneFit.test(df_te, previewHelper_te, previewType)  # 1st column: prediction (10) index ahead
    pdb.set_trace()

    true = np.array([carSpeed[idx:idx+predict_length] for idx in range(startIdx_test+predict_length, finalIdx_test)])


    fig = plt.figure()
    plt.plot(true[:, -1], '.-', label='True', linewidth=2)
    plt.plot(predict, '.-', label='Predict', linewidth=2)
    plt.legend()

    rmse = np.sqrt(sum((true[:, -1] - predict)**2)/true.shape[0])
    mape = 100*sum(abs(true[:, -1] - predict)/true[:, -1])/true.shape[0]
    plt.title('{}s ahead (RMSE:{:.3f}, MAPE:{:.3f})'.format((predict_length)/10, rmse, mape))
    plt.xlabel('Time [0.1s]')
    plt.ylabel('GPS_Speed')
    plt.show()

    # pdb.set_trace()



# predict_length에 따른 rmse 변화. 10까지는 급격하게 줄어듬. 10-40은 크게 변화 없고, 그 이후 다시 증가.
# (다시 증가하는 이유는? 매 시각에서 필요 이상으로 긴 시간동안의 preview는 현재의 속도 변화에 필요하지 않다?정도로 해석)
# rmses = []
# mapes = []
# for i in range(true.shape[1]-1):  # preview_distance가 50이었어서, k_preview의 [1:]부터 세면, 최대 길이는 49임.
#     rmse = np.sqrt(sum((true[:, i] - predict[:, i])**2)/true.shape[0])
#     rmses.append(rmse)
#     mape = 100*sum(abs(true[:, i] - predict[:, i])/true[:, i])/true.shape[0]
#     mapes.append(mape)
# plt.figure()
# plt.plot(rmses, '.-', linewidth=2, label='RMSE')
# plt.plot(mapes, '.-', linewidth=2, label='MAPE')
# plt.xlabel('Prediction length [0.1s]')
# plt.ylabel('Prediction Error (RMSE, MAPE)')
# plt.legend()
# plt.show()
#
# pdb.set_trace()

# for preview_distance in [600]:  #300, 900, 1200, 1800]:
#     # preview_distance = 600  # 10:1 = 600:60 1min.
#     previewHelper.set_preview_distance(preview_distance)  # 1:0.1 = 600:60  # 1 min.
#
#     corrs = []
#     for medfilt in range(5, 37, 2):  # 한번씩 계산만 하고 끝낼 거니까, generator를 쓰면 더 좋은 코드가 될 거 같긴함.
#         print(medfilt)
#         ks = np.array([])
#         vs = np.array([])
#         for idx in range(startIdx, finalIdx):
#             lat_preview, lon_preview = previewHelper.get_preview_plane(idx)
#             dist_preview, k_preview = previewHelper.get_preview_curve(idx, medfilt)
#
#             ks = np.append(ks, k_preview)
#             vs = np.append(vs, carSpeed[idx:idx+preview_distance])
#
#         corrs.append([medfilt, np.corrcoef(abs(ks), (-1)*vs)[0, 1]])  # pearsonr correlation
#
#     corrs = np.array(corrs)
#     # plt.plot(corrs[:, 0], corrs[:, 1])
#     # plt.show()
#     np.save('corrs_{}.npy'.format(preview_distance), corrs)
#     print(preview_distance, 'NPY SAVED')

# plt.figure()
# for preview_distance in [10, 100, 300, 600, 900, 1200, 1800]:
#     corrs = np.load('corrs_{}.npy'.format(preview_distance))
#     plt.plot(corrs[:, 0], corrs[:, 1], 'o--', label='{}'.format(preview_distance))
#     plt.hold(True)
# plt.xlabel('Median Filter Factor')
# plt.ylabel('Correlation (k, v)')
# plt.legend(loc='lower right')
# plt.show()


# # std_xxx.csv는 파일별로 row개수가 다르네
