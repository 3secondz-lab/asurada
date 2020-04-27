
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

def files2df(DATA_PATH, files):
    df_list = []
    for file in files:
        filepath = os.path.join(DATA_PATH, file)
        assert os.path.exists(filepath), 'No filename {}'.format(file)
        df_list.append(pd.read_csv(filepath))
    return pd.concat(df_list)

def draw_result_graph_fitting(true, predict, curvature, previewType, predictLength, true_curv=None):
    fig, ax = plt.subplots()
    line1 = ax.plot(true, 'b.-', label='True', linewidth=2)
    line2 = ax.plot(predict, 'r.-', label='Predict', linewidth=2)

    ax1 = ax.twinx()
    line3 = ax1.plot([(-1)*x for x in curvature], 'k--', label='Curvature', linewidth=1)

    if true_curv is not None:
        line4 = ax.plot(true_curv, 'b--', label='True_currv', linewidth=1)
        lines = line1 + line2 + line3 + line4
    else:
        lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs)

    rmse = np.sqrt(sum((true - predict)**2)/true.shape[0])
    mape = 100*sum(abs(true - predict)/true)/true.shape[0]

    if previewType == 'TIME':
        ax.set_title('{}s ahead (RMSE:{:.3f}, MAPE:{:.3f})'.format(predictLength, rmse, mape))
    elif previewType == 'DISTANCE':
        pass
    ax.set_xlabel('Time [0.1s]')
    ax.set_ylabel('GPS_Speed')
    ax1.set_ylabel('(-1)*|Curvature|')
    plt.show()
