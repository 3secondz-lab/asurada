
import pdb

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from helper import DataHelper

import Model
from Utils import *
from GraphUtils import *

model1 = True
model2 = True

''' Path Setting (Data, Model) '''
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'DATA')


''' Training/Test Data Setting
    # (developing...) You can use multiple record files to make one train/test data.
        - std_xxx.csv: recorded along with time [0.05s]
        - pos_xxx.csv: recorded along with distance [1m]
    * You can use the functions of GraphUtils to visualize data.
        - AnimatedPlots(df, previewType)
'''
datafiles_tr = ['std_001.csv']
df_tr = files2df(DATA_PATH, datafiles_tr)
df_tr_name = 'std-01-01'

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

preview_distance = 200  # unit: [m]
previewHelper_tr.set_preview_distance(preview_distance)
previewHelper_te.set_preview_distance(preview_distance)


''' ######################################## '''
''' Speed prediction using preview curvature '''
''' ######################################## '''

''' [Model1]: Using GPS_Speed: v(t+1) = f(k_(t+1)) '''
if model1:
    PolyFit = Model.PolyFit(df_tr, df_tr_name, previewHelper_tr, previewType,
                                order=7, xlabel='abs(k)', ylabel='(-1)*v', vis=1)

    carSpeed = df_te['GPS_Speed'].values
    predictLength = 1  # predict the speed at 1s ahead
    predict, abs_curvature, last_valid_idx = PolyFit.test(df_te, previewHelper_te, previewType, predictLength)

    if previewType == 'TIME':
        true = carSpeed[predictLength:predictLength+last_valid_idx+1]
    elif previewType == 'DISTANCE':
        pass  # ...

    # Graph
    draw_result_graph_fitting(true, predict, abs_curvature, previewType, predictLength)


''' [Model2]: Using Diff(GPS_Speed): v(t+1) = v(t) + f(k_(t+1:t+k), v_(t))
    Diff가 제대로 fitting되지 않고 있음... '''
if model2:
    predictLength = 1  # predict the speed at Xs ahead
    PlaneFit = Model.PlaneFit(df_tr, df_tr_name, previewHelper_tr, previewType, predictLength,
        order=2, xlabel='(-1)*speed', ylabel='feature of abs(k)', zlabel='speedDiff', vis=1)

    carSpeed = df_te['GPS_Speed'].values
    predict, abs_curvature_ft, last_valid_idx = PlaneFit.test(df_te, previewHelper_te, previewType)

    if previewType == 'TIME':
        true = carSpeed[predictLength*20:predictLength*20+last_valid_idx+1]
        true_curv = carSpeed[:last_valid_idx+1]
    elif previewType == 'DISTANCE':
        pass  # ...

    # Graph
    draw_result_graph_fitting(true, predict, abs_curvature_ft, previewType, predictLength, true_curv)
