import pdb

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from helper import DataHelper

import time

class xylim:
    def __init__(self, values):
        self.min_ = np.min(values)
        self.max_ = np.max(values)

    def update(self, values):
        if np.min(values) < self.min_:
            self.min_ = np.min(values)

        if np.max(values) > self.max_:
            self.max_ = np.max(values)

class AnimatedPlots:  # AnimatedScatter + AnimatedSubplots
    def __init__(self, df, startIdx=0, finalIdx=None,
                preview_distance=250, preview_time=10,
                prev_show_rate = 0.2, previewType='DISTANCE'):
        self.df = df
        self.startIdx = startIdx
        self.finalIdx = finalIdx

        if finalIdx is None:
            self.finalIdx = len(df)
        else:
            self.finalIdx = finalIdx

        assert previewType == 'DISTANCE' or previewType == 'TIME', 'Valid previewType: "DISTANCE" or "TIME"'

        self.previewType = previewType

        self.previewHelper = DataHelper(self.df)

        ''' Distance based preview '''
        self.preview_distance = preview_distance  # unit [m]
        self.previewHelper.set_preview_distance(preview_distance)

        ''' Time based preview '''
        self.preview_time = preview_time  # unit [s]
        self.previewHelper.set_preview_time(preview_time)

        self.psr = prev_show_rate
        # 더 긴 시간동안의 preview를 보아야, k와 speed의 correlation이 잘 나타나므로,
        # 이를 보이기위해, 긴 시간의 preview를 가져오지만,
        # driver model의 input으로는, psr만큼만의 preview를 사용하게 됨.
        # (preview가 길다고 무조건 좋은 건 아님.)

        self.stream = self.data_stream()

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(30, 8))
        '''
            ax1: localX/Y
            ax2: previewX/Y
            ax3: curvature vs. speed
        '''
        time.sleep(3)

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=3,
                                        init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        previewX, previewY, dist, k, speed, localX, localY = self.stream.__next__()

        self.prev_show = int(len(previewX) * self.psr)

        self.scat = self.ax1.scatter(localX[:-self.prev_show],
                                     localY[:-self.prev_show], alpha=0.3, color='k')
        self.scat_prev = self.ax1.scatter(localX[-self.prev_show:],
                                          localY[-self.prev_show:], alpha=0.3, color='r')

        ''' line2:
            the direction of the vehicle : +x
            -> so as to set the direction of the vehicle as +y, switch x, y like the below line
        '''
        self.line2 = self.ax2.plot(previewY[:self.prev_show],
                                   previewX[:self.prev_show], color='r', linewidth=2)

        self.line3, = self.ax3.plot(dist, k, color='r', linewidth=2)  # curvature

        self.ax3a = self.ax3.twinx()
        self.line4, = self.ax3a.plot(dist, k, color='b', linewidth=2)

        # set xylim
        self.xlim1 = xylim(self.df['PosLocalX'])
        self.ylim1 = xylim(self.df['PosLocalY'])

        self.xlim2 = xylim(previewY[:self.prev_show])
        self.ylim2 = xylim(previewX[:self.prev_show])

        self.ylim3 = xylim(k)
        self.ylim3a = xylim(speed)

        self.text = self.ax3.text(dist[int(len(dist)*0.5)], self.ylim3.max_,
                    'corr: {:.3}'.format(np.corrcoef(k, speed)[0, 1]),
                    fontsize=20)

        # set axis
        self.ax1.axis([self.xlim1.min_, self.xlim1.max_, self.ylim1.min_, self.ylim1.max_])
        self.ax1.set_xlabel('localX')
        self.ax1.set_ylabel('localY')
        self.ax1.set_title('Global Trajectory with Preview')

        self.ax2.axis([self.xlim2.min_, self.xlim2.max_, self.ylim2.min_, self.ylim2.max_])
        self.ax2.set_xlabel('previewY')
        self.ax2.set_ylabel('previewX')
        if self.previewType == 'DISTANCE':
            self.ax2.set_title('Preview Trajectory ({}m)'.format(self.prev_show))
        else:
            self.ax2.set_title('Preview Trajectory ({}s)'.format(self.prev_show/20))  # std.csv: unit [0.05s]

        self.ax3.set_ylim(self.ylim3.min_, self.ylim3.max_)
        self.ax3a.set_ylim(self.ylim3a.min_, self.ylim3a.max_)
        self.ax3.set_xlabel('Distance')
        self.ax3.set_ylabel('|Curvature|', color='r')
        if self.previewType == 'DISTANCE':
            self.ax3.set_title('Preview Curvature vs. Speed ({}m)'.format(self.preview_distance))
        else:
            self.ax3.set_title('Preview Curvature vs. Speed ({}s)'.format(self.preview_time))
        # Preview curvature vs. speed 이긴 하지만,
        # 사실상 이 그래프를 이루는 value들은, 주행 기록에서 계산된 것이므로,
        # 달려왔던 길에서의 curvature와 speed의 correlation을 보여서,
        # 이 기록을 기반으로 예측을 한다는 메세지를 전달해야 할 것 깉긴한데,
        # 그래도, 현재 이 코드처럼 preview의 curvature, speed를 보이고,
        # 여기 보이는 speed를 예측하는 것이 목표라고 설명하는 것으로 함.

        self.ax3a.set_ylabel('(-1)*GPS_Speed', color='b')

        def color_y_axis(ax, color):
            for t in ax.get_yticklabels():
                t.set_color(color)

        color_y_axis(self.ax3, 'r')
        color_y_axis(self.ax3a, 'b')

        for ax in [self.ax1, self.ax2, self.ax3, self.ax3a]:
            ax.grid()

        return self.scat, self.line2[0], self.line3, self.line4, self.text,

    def data_stream(self):
        for idx in range(self.startIdx, self.finalIdx):
            preview = self.previewHelper.get_preview(idx, method=self.previewType)  # dict
            previewX = preview['PreviewX']
            previewY = preview['PreviewY']
            dist = preview['Distance']
            k = preview['Curvature']
            speed = preview['GPS_Speed']

            self.prev_show = int(len(previewX) * self.psr)

            localX = self.df['PosLocalX'].iloc[self.startIdx:idx+self.prev_show].values
            localY = self.df['PosLocalY'].iloc[self.startIdx:idx+self.prev_show].values

            print('{}/{}'.format(idx, self.finalIdx), end='\r')  # just for checking
            yield previewX, previewY, dist, abs(k), (-1)*speed, localX, localY
        print('{}/{}'.format(idx, self.finalIdx))

    def update(self, i):
        previewX, previewY, dist, k, speed, localX, localY = self.stream.__next__()

        self.prev_show = int(len(previewX) * self.psr)

        self.scat.set_offsets(np.c_[localX[:-self.prev_show], localY[:-self.prev_show]])
        self.scat_prev.set_offsets(np.c_[localX[-self.prev_show:], localY[-self.prev_show:]])

        self.line2[0].set_data(previewY[:self.prev_show],
                               previewX[:self.prev_show])

        self.line3.set_data(dist, k)

        self.line4.set_data(dist, speed)

        self.text.set_text('corr: {:.3f}'.format(np.corrcoef(k, speed)[0, 1]))

        self.xlim2.update(previewY)
        self.ylim2.update(previewX)

        self.ax2.set_xlim(self.xlim2.min_, self.xlim2.max_)
        self.ax2.set_ylim(self.ylim2.min_, self.ylim2.max_)
        if self.previewType == 'DISTANCE':
            self.ax2.set_title('Preview Trajectory ({}m)'.format(self.prev_show))
        else:
            self.ax2.set_title('Preview Trajectory ({}s)'.format(self.prev_show/20))

        self.ylim3.update(k)
        self.ylim3a.update(speed)

        self.ax3.set_ylim(self.ylim3.min_, self.ylim3.max_)
        self.ax3a.set_ylim(self.ylim3a.min_, self.ylim3a.max_)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax3a]:
            ax.grid()

        return self.scat, self.line2[0], self.line3, self.line4, self.text,

class AnimatedSubplots:
    def __init__(self, df, startIdx=0, finalIdx=None,
                preview_distance=250, preview_time=10, previewType='DISTANCE'):
        self.df = df
        self.startIdx = startIdx
        self.finalIdx = finalIdx

        if finalIdx is None:
            self.finalIdx = len(df)
        else:
            self.finalIdx = finalIdx

        assert previewType == 'DISTANCE' or previewType == 'TIME', 'Valid previewType: "DISTANCE" or "TIME"'

        self.previewType = previewType

        self.previewHelper = DataHelper(self.df)

        ''' Distance based preview '''
        self.preview_distance = preview_distance  # unit [m]
        self.previewHelper.set_preview_distance(preview_distance)

        ''' Time based preview '''
        self.preview_time = preview_time  # unit [s]
        self.previewHelper.set_preview_time(preview_time)

        self.stream = self.data_stream()

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 8))

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                        init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        previewX, previewY, dist, k, speed = self.stream.__next__()

        '''
            the direction of the vehicle : +x
            -> so as to set the direction of the vehicle as +y, switch x, y like the below line
        '''

        self.line1 = self.ax1.plot(previewY, previewX)  # The direction of the vehicle: +x
        self.line2, = self.ax2.plot(dist, k, color='r')  # curvature

        self.ax2a = self.ax2.twinx()
        self.line3, = self.ax2a.plot(dist, speed, color='b')

        # set axis
        self.xlim1 = xylim(previewY)
        self.ylim1 = xylim(previewX)

        self.ax1.set_xlim(self.xlim1.min_, self.xlim1.max_)
        self.ax1.set_ylim(self.ylim1.min_, self.ylim1.max_)
        self.ax1.set_xlabel('previewY')
        self.ax1.set_ylabel('previewX')
        if self.previewType == 'DISTANCE':
            self.ax1.set_title('Preview Trajectory ({}m)'.format(self.preview_distance))
        else:
            self.ax1.set_title('Preview Trajectory ({}s)'.format(self.preview_time))

        self.ylim2 = xylim(k)
        self.ylim2a = xylim(speed)

        self.ax2.set_ylim(self.ylim2.min_, self.ylim2.max_)
        self.ax2a.set_ylim(self.ylim2a.min_, self.ylim2a.max_)
        self.ax2.set_xlabel('Distance')
        self.ax2.set_ylabel('|Curvature|', color='r')
        self.ax2a.set_ylabel('(-1)*GPS_Speed', color='b')

        def color_y_axis(ax, color):
            for t in ax.get_yticklabels():
                t.set_color(color)

        color_y_axis(self.ax2, 'r')
        color_y_axis(self.ax2a, 'b')

        for ax in [self.ax1, self.ax2, self.ax2a]:
            ax.grid()

        return self.line1[0], self.line2, self.line3,

    def data_stream(self):
        for idx in range(self.startIdx, self.finalIdx):
            preview = self.previewHelper.get_preview(idx, method=self.previewType)  # dict
            previewX = preview['PreviewX']
            previewY = preview['PreviewY']
            dist = preview['Distance']
            k = preview['Curvature']
            speed = preview['GPS_Speed']
            print('{}/{}'.format(idx, self.finalIdx), end='\r')  # just for checking
            yield previewX, previewY, dist, abs(k), (-1)*speed
        print('{}/{}'.format(idx, self.finalIdx))

    def update(self, i):
        previewX, previewY, dist, k, speed = self.stream.__next__()

        self.line1[0].set_data(previewY, previewX)
        self.line2.set_data(dist, k)
        self.line3.set_data(dist, speed)

        self.xlim1.update(previewY)
        self.ylim1.update(previewX)

        self.ax1.set_xlim(self.xlim1.min_, self.xlim1.max_)
        self.ax1.set_ylim(self.ylim1.min_, self.ylim1.max_)

        self.ylim2.update(k)
        self.ylim2a.update(speed)

        self.ax2.set_ylim(self.ylim2.min_, self.ylim2.max_)
        self.ax2a.set_ylim(self.ylim2a.min_, self.ylim2a.max_)

        for ax in [self.ax1, self.ax2, self.ax2a]:
            ax.grid()

        return self.line1[0], self.line2, self.line3,

class AnimatedScatter:
    def __init__(self, df, startIdx=0, finalIdx=None):
        self.df = df
        self.startIdx = startIdx

        if finalIdx is None:
            self.finalIdx = len(df)
        else:
            self.finalIdx = finalIdx

        assert sum([0 if x in df.columns else 1 for x in
            ['PosLocalX', 'PosLocalY']]) == 0, 'Data should contain "PosLocalX" and "PosLocalY" fields'
        # from PosLon, PosLat

        self.stream = self.data_stream()

        self.fig, self.ax = plt.subplots()

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                        init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        data = self.stream.__next__()
        self.scat = self.ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        self.scat2 = self.ax.scatter(data[-1:, 0], data[-1:, 1], color='r')
        xlim = xylim(self.df['PosLocalX'])
        ylim = xylim(self.df['PosLocalY'])
        self.ax.axis([xlim.min_, xlim.max_, ylim.min_, ylim.max_])
        self.ax.set_xlabel('localX')
        self.ax.set_ylabel('localY')
        self.ax.set_title('Global Trajectory')
        return self.scat, self.scat2 # iterable

    def data_stream(self):
        for i in range(self.startIdx, self.finalIdx):
            localX = self.df['PosLocalX'].iloc[self.startIdx:i].values
            localY = self.df['PosLocalY'].iloc[self.startIdx:i].values
            print('{}/{}'.format(i, self.finalIdx), end='\r')
            yield np.c_[localX, localY]  # (# rows, 2)
        print('{}/{}'.format(i, self.finalIdx))

    def update(self, i):
        data = self.stream.__next__()
        self.scat.set_offsets(data[:, :2])
        self.scat2.set_offsets(data[-1:, :2])
        return self.scat, self.scat2

if __name__ == "__main__":
    ''' GraphUtils Example Code
        (datafile: std_xxx.csv (Record in time order),
                   pos_xxx.csv (Record in location order)) '''

    # df = pd.read_csv('std_001.csv')
    df = pd.read_csv('pos_001.csv')

    # df = pd.read_csv('mkkim-recoder-scz_msgs.csv')  # 현재 지원안됨.
    # df = pd.read_csv('sdpark-recoder-scz_msgs.csv')  # 현재 지원안됨.

    ''' 주행 궤적 기록 애니메이션 '''
    # ani = AnimatedScatter(df)
    # plt.show()

    ''' 주행 preview & curvature-GPS_SPeed 기록 애니메이션 '''
    # ani = AnimatedSubplots(df, previewType='DISTANCE')
    # ani = AnimatedSubplots(df, previewType='TIME')
    # plt.show()

    ''' 주행 궤적 기록 + 주행 preview & Curvature-Speed 기록 애니메이션 '''
    ani = AnimatedPlots(df, previewType='DISTANCE')
    # ani = AnimatedPlots(df, previewType='TIME')
    plt.show()
