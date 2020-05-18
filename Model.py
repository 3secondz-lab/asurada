import pdb

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

from Utils import *

class PlaneFit:
    def __init__(self, df, df_name, previewHelper, previewType, predictLeng, recFreq,
                        order, xlabel=None, ylabel=None, zlabel=None, vis=False):

        assert order == 1 or order == 2, 'Order has to be 1 or 2'
        self.order = order

        self.previewHelper = previewHelper
        preview_time = previewHelper.preview_time
        preview_distance = previewHelper.preview_distance

        self.predictLeng = predictLeng

        if previewType == 'TIME':
            dataset4fit = 'ks_{}_{}s.npy'.format(df_name, preview_time)
        elif previewType == 'DISTANCE':
            dataset4fit = 'ks_{}_{}m.npy'.format(df_name, preview_distance)

        if not os.path.isfile(dataset4fit):
            ks = buildDataset4fit(df, previewHelper, previewType)
            np.save(dataset4fit, ks)
        else:
            ks = np.load(dataset4fit)

        # ks = np.load(dataset4fit)
        vs = df['GPS_Speed'].values

        vsdiff = vs[predictLeng*recFreq:] - vs[:-predictLeng*recFreq]

        vs = vs[:-predictLeng*recFreq]  # X1

        self.k_preview_idx = 3
        ks = ks[:-predictLeng*recFreq, self.k_preview_idx:]  # X2
        # 초반에 나오는 curvature는 거의 0에 가까워서 제외.
        ## record 데이터에서는 ks가 -1, -2 데이터가 nan임... 왜 그렇지...
        # (사실상 평균이라기 보다 median임., previewType이 시간이고, data도 시간으로 정렬되어 있어도 preview curvature의 길이가 달라지나봄.)

        # 평균 preview 길이 구하기.
        avgLenPreview = int(np.median(np.sum(np.isnan(ks), axis=1)))
        ks = ks[:, :-avgLenPreview]

        ## 이 부분도, 이 관계를 다시 찾아야 할지도 모름...,
        ## NN으로 가면 abs나 (-1) 없이 raw data를 그대로 쓸 수 있을지도...
        ks_transformed = abs(ks)
        vs_transformed = (-1)*vs

        # pdb.set_trace()

        # remove nan value
        ## track 한바퀴가 거의 끝나갈 쯤이라, 충분한 preview가 생성되지 않는 경우.
        notNaNidx = np.where(np.sum(np.isnan(ks_transformed), axis=1)==0)
        ks_transformed = ks_transformed[notNaNidx]
        vs_transformed = vs_transformed[notNaNidx]
        vsdiff = vsdiff[notNaNidx]

        # k-feature 1.  # 매 preview의 curvature list의 길이가 달라져서, 아래 코드로 작성된 PCA는 지원 안됨.
        # pca = PCA(n_components=1)
        # ks_low = pca.fit_transform(ks_transformed)

        # k-feature 2.
        # ks_mean = np.nanmean(ks_transformed, axis=1)

        # k-feature 3. (raw data at predictLeng aheads)
        ks_raw = ks_transformed[:, -1]

        # normalization
        self.vs_norm = dataNormalization(vs_transformed)
        # self.ks_low_norm = dataNormalization(ks_low)  # pca, 지원안됨
        # self.ks_mean_norm = dataNormalization(ks_mean)
        self.ks_raw_norm = dataNormalization(ks_raw)

        # Planefit model
        # data = np.c_[self.vs_norm.data, self.ks_low_norm.data, vsdiff]  # pca, 지원안됨
        # data = np.c_[self.vs_norm.data, self.ks_mean_norm.data, vsdiff]
        data = np.c_[self.vs_norm.data, self.ks_raw_norm.data, vsdiff]

        X, Y = np.meshgrid(np.arange(-3, 3, 0.5), np.arange(-4, 4, 0.5))  # for vis

        if order == 1:
            A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
            self.C,_,_,_ = scipy.linalg.lstsq(A, data[:, 2])
            Z = self.C[0]*X + self.C[1]*Y + self.C[2]

        elif order == 2:
            A = np.c_[data[:, 0]**2, data[:, 0], data[:, 1]**2, data[:, 1], np.ones(data.shape[0])]
            self.C,_,_,_ = scipy.linalg.lstsq(A, data[:, 2])
            Z = self.C[0]*(X**2) + self.C[1]*X + self.C[2]*(Y**2) + self.C[3]*Y + self.C[4]

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.scatter(data[:, 0], data[:, 1], alpha=0.3)
        # ax2.scatter(data[:, 0], data[:, 2], alpha=0.3)
        # ax3.scatter(data[:, 1], data[:, 2], alpha=0.3)
        #
        # ax1.set_xlabel('(-1)*speed')
        # ax1.set_ylabel('feature of abs(k)')
        #
        # ax2.set_xlabel('(-1)*speed')
        # ax2.set_ylabel('speedDiff')
        #
        # ax3.set_xlabel('feature of abs(k)')
        # ax3.set_ylabel('speedDiff')
        #
        # plt.show()
        #
        # pdb.set_trace()

        if vis:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.show()

    def test(self, df, previewHelper, previewType, recFreq):

        assert previewType == 'TIME', 'previewType (DISTANCE) is not currently supported.'

        # predict the speed at predictLeng (s or m) ahead
        if previewType == 'TIME':
            previewHelper.set_preview_time(self.predictLeng)
        elif previewType == 'DISTANCE':
            previewHelper.set_preview_distance(self.predictLeng)

        curvature = []  # abs
        predict = []

        last_valid_idx = -1

        for idx in range(len(df)):
            print('Predicting... @ origin idx {}/{}'.format(idx, len(df)), end='\r')

            v = df['GPS_Speed'].iloc[idx]  # current speed
            v_norm = ((-1)*v - self.vs_norm.mu)/self.vs_norm.std

            preview = previewHelper.get_preview(idx, previewType)

            if previewType == 'TIME':
                ks = preview['Curvature']
                # pdb.set_trace()
                if len(ks) > self.predictLeng * recFreq:
                    ks = ks[:self.predictLeng * recFreq]
                if len(ks) == self.predictLeng * recFreq:
                    k = np.mean(abs(ks[self.k_preview_idx:]))
                    # k_norm = (k - self.ks_mean_norm.mu)/self.ks_mean_norm.std
                    k_norm = (k - self.ks_raw_norm.mu)/self.ks_raw_norm.std

                    if self.order == 1:
                        vdiff = self.C[0] * v_norm + self.C[1] * k_norm + self.C[2]
                    elif self.order == 2:
                        vdiff = self.C[0]*(v_norm**2) + self.C[1]*v_norm + self.C[2]*(k_norm**2) + self.C[3]*k_norm + self.C[4]

                    curvature.append(k)
                    predict.append(v+vdiff)
                    last_valid_idx = idx
                else:
                    break
            elif previewType == 'DISTANCE':
                pass
        print('Predicting... @ origin idx {}/{}'.format(idx, len(df)))
        return np.array(predict), np.array(curvature), last_valid_idx

class PolyFit:
    def __init__(self, df, df_name, previewHelper, previewType, order,
                                        xlabel=None, ylabel=None, vis=False):
        self.previewHelper = previewHelper

        preview_time = previewHelper.preview_time
        preview_distance = previewHelper.preview_distance

        # Data Loading (k vs. v)
        if previewType == 'TIME':
            dataset4fit = 'ks_{}_{}s.npy'.format(df_name, preview_time)
        elif previewType == 'DISTANCE':
            dataset4fit = 'ks_{}_{}m.npy'.format(df_name, preview_distance)

        if not os.path.isfile(dataset4fit):
            ks = []
            vs = []
            for idx in range(len(df)):
                print('Building training set... {}/{}'.format(idx, len(df)), end='\r')

                preview = previewHelper.get_preview(idx, previewType)
                ks.append(preview['Curvature'])
                vs.append(preview['GPS_Speed'])

            pad = len(max(ks, key=len))
            ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])
            vs_arr = np.array([v.tolist() + [np.nan]*(pad-len(v)) for v in vs])

            np.save(dataset4fit, ks_arr)
            np.save('vs'+dataset4fit[2:], vs_arr)

        ks = np.load(dataset4fit)  # (#of rows of dataset, maximum length of preview)
        vs = np.load('vs'+dataset4fit[2:])

        # Additional Preprocessing
        # preview를 계산하는 매 origin에서의 curvature는 거의 0으로 계산되는데,
        # 이 때문에 polyfit을 수행하기 위해 (k, v)들을 모았을 때, curvature가 0인 부분에서
        # 다양한 속력 값이 나타난다. 따라서, polyfit시 앞 몇 포인트는 제외하는 것으로 처리.
        ks = ks[:, 3:]
        vs = vs[:, 3:]

        ks = ks.flatten()  # to remove nan value
        vs = vs.flatten()

        ks = ks[~np.isnan(ks)]
        vs = vs[~np.isnan(vs)]

        x = abs(ks)
        y = (-1)*vs

        # Coefficient check
        print('Coefficient Check')
        print('{} vs {}:{}'.format(xlabel, ylabel, np.corrcoef(x, y)[0, 1]))

        # Polyfit model
        self.order = order
        self.params = np.polyfit(x, y, order)

        if vis:
            plt.figure()
            appx = [*np.linspace(np.min(x), np.max(x), 100)]
            appy = [self.params[-1]] * len(appx)
            for i in range(order):  # k=2 -> 0:**2, 1:**1, 2:**0
                appy = [yy+self.params[i]*(xx**(order-i)) for xx, yy in zip(appx, appy)]
            plt.plot(appx, appy, 'r--', linewidth=2)
            # plt.hold(True)
            plt.scatter(x, y, alpha=0.1)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

    def test(self, df, previewHelper, previewType, predictLeng, recFreq):
        # predict the speed at predictLeng (s or m) ahead

        assert previewType == 'TIME', 'previewType (DISTANCE) is not currently supported.'

        if previewType == 'TIME':
            previewHelper.set_preview_time(predictLeng)
        elif previewType == 'DISTANCE':
            previewHelper.set_preview_distance(predictLeng)

        curvature = []
        predict = []

        last_valid_idx = -1

        # pdb.set_trace()
        for idx in range(len(df)):
            print('Predicting... @ origin idx {}/{}'.format(idx, len(df)), end='\r')

            preview = previewHelper.get_preview(idx, previewType)

            if previewType == 'TIME':
                if len(preview['Curvature']) >= predictLeng * recFreq:  # unit: 0.05s
                    k = abs(preview['Curvature'][-1])
                    v = self.params[-1]

                    for i in range(self.order):
                        v += self.params[i]*(k**(self.order-i))

                    curvature.append(k)
                    predict.append((-1)*v)

                    last_valid_idx = idx
                else:
                    break

            elif previewType == 'DISTANCE':
                pass
        print('Predicting... @ origin idx {}/{}'.format(idx, len(df)))
        return np.array(predict), np.array(curvature), last_valid_idx
