import pdb

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

from Utils import *

class PlaneFit:
    # def __init__(self, startIdx, finalIdx, df, previewHelper, predict_length, order,
    #                         xlabel=None, ylabel=None, zlabel=None, vis=False):
    def __init__(self, df, df_name, previewHelper, previewType, predictLength,
                        order, xlabel=None, ylabel=None, zlabel=None, vis=False):

        assert order == 1 or order == 2, 'Order has to be 1 or 2'
        self.order = order

        # self.df = df

        self.previewHelper = previewHelper
        preview_time = previewHelper.preview_time
        preview_distance = previewHelper.preview_distance

        self.predictLength = predictLength

        if previewType == 'TIME':
            dataset4fit = 'ks_{}_{}s.npy'.format(df_name, preview_time)
        elif previewType == 'DISTANCE':
            dataset4fit = 'ks_{}_{}m.npy'.format(df_name, preview_distance)

        # if not os.path.isfile('ks_{}_{}_{}.npy'.format(startIdx, finalIdx, preview_distance)):
        if not os.path.isfile(dataset4fit):
            ks = []
            # vs = []
            # for idx in range(startIdx, finalIdx):
            for idx in range(len(df)):
                # dist_preview, k_preview = previewHelper.get_preview_curve(idx, medfilt=15)  # default=15
                # ks.append(k_preview)
                # vs.append(df['GPS_Speed'].iloc[idx:idx+preview_distance].values)

                preview = previewHelper.get_preview(idx, previewType)
                ks.append(preview['Curvature'])
            # ks = np.array(ks)

            pad = len(max(ks, key=len))
            ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])

            # vs = np.array(vs)

            # np.save('ks_{}_{}_{}.npy'.format(startIdx, finalIdx, preview_distance), ks)
            # np.save('vs_{}_{}_{}.npy'.format(startIdx, finalIdx, preview_distance), vs)
            np.save(dataset4fit, ks_arr)

        # ks = np.load('ks_{}_{}_{}.npy'.format(startIdx, finalIdx, preview_distance))
        # vs = np.load('vs_{}_{}_{}.npy'.format(startIdx, finalIdx, preview_distance))
        ks = np.load(dataset4fit)


        # vs = df['GPS_Speed'].iloc[startIdx:finalIdx].values
        vs = df['GPS_Speed'].values
        # pdb.set_trace()
        # vsdiff = np.diff(vs)  # Y  # 여기를 predict_length를 고려해서 diff를 계산해야 하나?

        vsdiff = vs[predictLength:] - vs[:-predictLength]

        vs = vs[:-predictLength]  # X1

        self.k_preview_idx = 3
        ks = ks[:-predictLength, self.k_preview_idx:]  # X2  # 초반에 나오는 curvature는 거의 0에 가까워서 제외.

        ks_transformed = abs(ks)
        vs_transformed = (-1)*vs

        # k-feature 1.  # 매 preview의 curvature list의 길이가 달라져서, 현재 pca 아래 방법으로는 지원 안됨.
        # pca = PCA(n_components=1)
        # ks_low = pca.fit_transform(ks_transformed)

        # k-feature 2.
        # ks_mean = np.mean(ks_transformed, axis=1)
        ks_mean = np.nanmean(ks_transformed, axis=1)

        # normalization
        self.vs_norm = dataNormalization(vs_transformed)
        # self.ks_low_norm = dataNormalization(ks_low)
        self.ks_mean_norm = dataNormalization(ks_mean)

        # Planefit model
        # data = np.c_[self.vs_norm.data, self.ks_low_norm.data, vsdiff]  # (rows, 3)
        data = np.c_[self.vs_norm.data, self.ks_mean_norm.data, vsdiff]  # (rows, 3)

        X, Y = np.meshgrid(np.arange(-3, 3, 0.5), np.arange(-4, 4, 0.5))  # for vis

        if order == 1:
            A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
            self.C,_,_,_ = scipy.linalg.lstsq(A, data[:, 2])
            Z = self.C[0]*X + self.C[1]*Y + self.C[2]

        elif order == 2:
            A = np.c_[data[:, 0]**2, data[:, 0], data[:, 1]**2, data[:, 1], np.ones(data.shape[0])]
            # A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2]**2]
            self.C,_,_,_ = scipy.linalg.lstsq(A, data[:, 2])

            # XX = X.flatten()
            # YY = Y.flatten()

            # Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY*2], self.C).reshape(X.shape)
            Z = self.C[0]*(X**2) + self.C[1]*X + self.C[2]*(Y**2) + self.C[3]*Y + self.C[4]

        if vis:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.show()

    # def test(self, startIdx_test, finalIdx_test):
    def test(self, df, previewHelper, previewType):
        predict = []

        last_valid_idx = -1

        for idx in range(len(df)):
            v = df['GPS_Speed'].iloc[idx]
            v_norm = ((-1)*v - self.vs_norm.mu)/self.vs_norm.std

            # dist_preview, k_preview = self.previewHelper.get_preview_curve(idx)
            preview = previewHelper.get_preview(idx, previewType)

            # k = np.mean(abs(k_preview[self.k_preview_idx:]))
            k_leng = abs(preview['Curvature'][self.k_preview_idx:])
            k = np.mean(k_leng)
            k_norm = (k - self.ks_mean_norm.mu)/self.ks_mean_norm.std

            if self.order == 1:
                vdiff = self.C[0] * v_norm + self.C[1] * k_norm + self.C[2]
            elif self.order == 2:
                vdiff = self.C[0]*(v_norm**2) + self.C[1]*v_norm + self.C[2]*(k_norm**2) + self.C[3]*k_norm + self.C[4]

            if len(k_leng) == self.predictLength:
                last_valid_idx = idx
                predict.append(v+vdiff)
        return np.array(predict), last_valid_idx  # (# of rows, 1)

from network import NetModel
from network import Agent

class PolyFit:
    def __init__(self, df, df_name, previewHelper, previewType,
                 input_size, h1, h2, output_size, lr, batch_size):
        self.previewHelper = previewHelper
        # self.agent = Agent(30, 30, 20, 10, lr=0.2, batch_size=4096)
        self.agent = Agent(input_size, h1, h2, output_size, lr, batch_size)
        self.input_size = input_size
        self.output_size= output_size
        self.lr = lr
        self.batch_size = batch_size
        self.previewType = previewType
        self.previewHelper = previewHelper

        self.preview_time = previewHelper.preview_time
        self.preview_distance = previewHelper.preview_distance

        # Data Loading (k vs. v)
        if previewType == 'TIME':
            self.dataset4fit = 'ks_{}_{}s.npy'.format(df_name, self.preview_time)
        elif previewType == 'DISTANCE':
            self.dataset4fit = 'ks_{}_{}m.npy'.format(df_name, self.preview_distance)


    def training(self):
        if not os.path.isfile(self.dataset4fit):
            ks = []
            vs = []
            for idx in range(len(df)):
                preview = self.previewHelper.get_preview(idx, self.previewType)
                ks.append(preview['Curvature'])
                vs.append(preview['GPS_Speed'])
            pad = len(max(ks, key=len))
            ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])
            vs_arr = np.array([v.tolist() + [np.nan]*(pad-len(v)) for v in vs])

            np.save(self.dataset4fit, ks_arr)
            np.save('vs'+self.dataset4fit[2:], vs_arr)
        pdb.set_trace()
        ks = np.load(self.dataset4fit)
        vs = np.load('vs'+self.dataset4fit[2:])

        # Additional Preprocessing
        ks = ks[:, 3:]  # origin에서의 curvature는 거의 항상 0이다. 그래서, 원래 그래프에서 curvature 0에서 다양한 속력이 나왔던  듯.
        vs = vs[:, 3:]  # 그래서 polyfit을 할 때에는, 조금 빼고 fit을 하는 걸로.

        x = ks.flatten()
        y = vs.flatten()

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        # x = abs(x)
        # y = (-1)*y

        x_mem = list()
        y_mem = list()

        for i in range(len(x)-self.input_size):
            x_mem.append(np.concatenate(([y[i]*10**(-6)],x[i:i+self.input_size-1])))
            # pdb.set_trace()
            y_mem.append(y[i:i+self.output_size])
        print(self.agent.update(x_mem, y_mem))

    def test(self, df, previewHelper, previewType):
        predict = []
        curvature = []
        true_vals = []

        last_valid_idx = -1

        for idx in range(len(df)-self.input_size-1):

            preview = previewHelper.get_preview(idx, previewType)

            # k_leng = abs(preview['Curvature'][1:1+input_size+1])
            k_leng = preview['Curvature'][1:98]
            k_leng = np.concatenate((k_leng, k_leng[1:4]))
            k_leng = np.concatenate(([preview['GPS_Speed'][1]*10**(-6)],k_leng))
            # pdb.set_trace()
            # v_leng = [self.params[-1]] * leng
            v_leng = self.agent.predictor(k_leng)

            curvature.append(k_leng)
            predict.append(v_leng.cpu().detach().numpy())
            true_vals.append(preview['GPS_Speed'][1:self.output_size+1])
            if idx%100==0:
                print(k_leng, v_leng)
            # for i in range(self.order):
            #     v_leng = [y + self.params[i]*(x**(self.order-i)) for x, y in zip(k_leng, v_leng)]
            # if len(v_leng) == leng:
            #     last_valid_idx = idx
            #     predict.append([(-1)*y for y in v_leng])
            #     curvature.append(k_leng.tolist())  # curvature 마다 길이가 다름.
        # return np.array(predict), curvature, last_valid_idx  # predict: (# of rows, leng)
        return np.array(predict), np.array(true_vals), curvature, len(predict)   # predict: (# of rows, leng)


    def test_old(self, df, previewHelper, previewType):
        predict = []
        curvature = []
        true_vals = []

        last_valid_idx = -1

        for idx in range(len(df)-self.input_size-1):

            preview = previewHelper.get_preview(idx, previewType)

            # k_leng = abs(preview['Curvature'][1:1+input_size+1])
            k_leng = preview['Curvature'][1:1+self.input_size]
            # k_leng = np.concatenate((k_leng, k_leng[1:4]))
            # k_leng = np.concatenate(([preview['GPS_Speed'][1]*10**(-6)],k_leng))
            # pdb.set_trace()
            # v_leng = [self.params[-1]] * leng
            v_leng = self.agent.predictor(k_leng)

            curvature.append(k_leng)
            predict.append(v_leng.cpu().detach().numpy())
            # true_vals.append(preview['GPS_Speed'][1:self.output_size+1])
            if idx%100==0:
                print(k_leng, v_leng)
            # for i in range(self.order):
            #     v_leng = [y + self.params[i]*(x**(self.order-i)) for x, y in zip(k_leng, v_leng)]
            # if len(v_leng) == leng:
            #     last_valid_idx = idx
            #     predict.append([(-1)*y for y in v_leng])
            #     curvature.append(k_leng.tolist())  # curvature 마다 길이가 다름.
        # return np.array(predict), curvature, last_valid_idx  # predict: (# of rows, leng)
        return np.array(predict), curvature, len(predict)   # predict: (# of rows, leng)