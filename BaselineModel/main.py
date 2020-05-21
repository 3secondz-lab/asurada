
import pdb

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import DataHelper

import Model
from Utils import *

model1 = True
model2 = True

''' Path Setting (Data, Model) '''
# current_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(current_dir, 'DATA')
DATA_PATH = '../Data'


''' Training/Test Data Setting
    # (developing...) You can use multiple record files to make one train/test data.
        - std_xxx.csv: recorded along with time [0.05s]
        - pos_xxx.csv: recorded along with distance [1m]
    * You can use the functions of GraphUtils to visualize data.
        - AnimatedPlots(df, previewType)
'''
## std_xxx.csv
# datafiles_tr = ['std_001.csv']
# df_tr = files2df(DATA_PATH, datafiles_tr)
# df_tr_name = 'std-01-01'
# recFreq = 20  # unit: 0.05s

# datafiles_te = ['std_002.csv']
# df_te = files2df(DATA_PATH, datafiles_te)

## mkkim/sdpark-recoder-scz_msgs.csv
trteRate = 0.7
datafiles_tr = ['mkkim-recoder-scz_msgs.csv']
df_tr = files2df(DATA_PATH, datafiles_tr)
df_tr_name = 'mkkim-{}%'.format(int(trteRate*100))
recFreq = 10

datafiles_te = ['mkkim-recoder-scz_msgs.csv']
df_te = files2df(DATA_PATH, datafiles_te)

df_tr = df_tr.iloc[:int(len(df_tr)*trteRate)]
df_te = df_te.iloc[int(len(df_te)*trteRate):]


''' Preview Helper for train/test '''
previewHelper_tr = DataHelper(df_tr)
previewHelper_te = DataHelper(df_te)  # preview needed when the test phase

previewType = 'TIME'
# previewType = 'DISTANCE'

preview_time = 10  # unit: [s]
previewHelper_tr.set_preview_time(preview_time)
previewHelper_te.set_preview_time(preview_time)

preview_distance = 200  # unit: [m]
previewHelper_tr.set_preview_distance(preview_distance)
previewHelper_te.set_preview_distance(preview_distance)


''' ######################################## '''
''' Speed prediction using preview curvature '''
''' ######################################## '''

''' [Model1]: Predict GPS_Speed: v(t+1) = f(k_(t+1)) '''
if model1:
    # PolyFit = Model.PolyFit(df_tr, df_tr_name, previewHelper_tr, previewType,
                                # order=7, xlabel='abs(k)', ylabel='(-1)*v', vis=1)  # std
    print('=== PolyFit ===')

    PolyFit = Model.PolyFit(df_tr, df_tr_name, previewHelper_tr, previewType, order=5, xlabel='abs(k)', ylabel='(-1)*v', vis=1)  # mkkim

    carSpeed = df_te['GPS_Speed'].values
    predictLength = 5  # predict the speed at 1s ahead
    predict, abs_curvature, last_valid_idx = PolyFit.test(df_te, previewHelper_te,
                                    previewType, predictLength, recFreq=recFreq)
    if previewType == 'TIME':
        true = carSpeed[predictLength:predictLength+last_valid_idx+1]
    elif previewType == 'DISTANCE':
        pass  # ...

    # Graph
    draw_result_graph_fitting(true, predict, abs_curvature, previewType, predictLength)
    draw_result_graph_fitting(true, predict, abs_curvature, previewType, predictLength, idxFrom=2500, idxTo=4500)

''' [Model2]: Predict Diff(GPS_Speed): v(t+1) = v(t) + f(k_(t+1:t+k), v_(t))
    * f(k_(t+1:t+k)) = k(t+k)
    Diff가 제대로 fitting되지 않고 있음... '''
if model2:
    print('=== PlaneFit ===')

    predictLength = 5  # predict the speed at Xs ahead
    PlaneFit = Model.PlaneFit(df_tr, df_tr_name, previewHelper_tr, previewType, predictLength, recFreq,
        order=2, xlabel='(-1)*speed', ylabel='feature of abs(k)', zlabel='speedDiff', vis=1)

    predict, abs_curvature_ft, last_valid_idx = PlaneFit.test(df_te, previewHelper_te,
                                                                previewType, recFreq)

    # True value
    carSpeed = df_te['GPS_Speed'].values
    if previewType == 'TIME':
        true = carSpeed[predictLength*recFreq:predictLength*recFreq+last_valid_idx+1]
        true_curv = carSpeed[:last_valid_idx+1]
    elif previewType == 'DISTANCE':
        pass  # ...

    if len(predict) != len(true):  # predictLength가 길어지면, true 길이가 짧아질 수 있음.
        predict = predict[:len(true)]
        abs_curvature_ft = abs_curvature_ft[:len(true)]
        true_curv = true_curv[:len(true)]

    # Graph
    draw_result_graph_fitting(true, predict, abs_curvature_ft, previewType, predictLength, true_curv)
    draw_result_graph_fitting(true, predict, abs_curvature_ft, previewType, predictLength, true_curv, idxFrom=2500, idxTo=4500)
