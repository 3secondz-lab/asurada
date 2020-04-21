import pdb

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from helper import DataHelper

import time

class AnimatedPlots:  # AnimatedScatter + AnimatedSubplots
    def __init__(self, df, startIdx, finalIdx, preview_distance=600, prev_dist_show=50):
        self.df = df
        self.startIdx = startIdx
        self.finalIdx = finalIdx

        self.previewHelper = DataHelper(lat=df['PosLat'].values,
                                        lon=df['PosLon'].values,
                                        heading=df['AngleTrack'].values)
        self.preview_distance = preview_distance
        self.previewHelper.set_preview_distance(self.preview_distance)

        self.prev_dist_show = prev_dist_show  # preview 부분은 좀더 짧아야 이해가 쉬움 (그래프 상에서).

        self.stream = self.data_stream()

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(30, 8))
        '''
            ax1: global position
            ax2: preview
            ax3: curvature vs. speed
        '''
        time.sleep(3)

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=3,
                                        init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        data, lat, lon = self.stream.__next__()

        '''
        lat_preview:    data[:, 0]
        lon_preview:    data[:, 1]
        dist_preview:   data[:, 2]
        abs(k_preview): data[:, 3]
        (-1)*carSpeed:  data[:, 4]
        posLat:         data[:, 5] -> lat (현재+prev_dist_show)
        posLon:         data[:, 6] -> lon (현재+prev_dist_show)
        '''

        self.scat = self.ax1.scatter(lon[:-self.prev_dist_show],
                                     lat[:-self.prev_dist_show], alpha=0.3, color='k')
        self.scat_prev = self.ax1.scatter(lon[-self.prev_dist_show:],
                                          lat[-self.prev_dist_show:], alpha=0.3, color='r')

        self.line2 = self.ax2.plot(data[:self.prev_dist_show, 1],
                                   data[:self.prev_dist_show, 0], color='r', linewidth=2)

        self.line3, = self.ax3.plot(data[:, 2], data[:, 3], color='r', linewidth=2)  # curvature

        self.ax3a = self.ax3.twinx()
        self.line4, = self.ax3a.plot(data[:, 2], data[:, 4], color='b', linewidth=2)  # something

        self.text = self.ax3.text(400, 0.04,
                    'corr: {}'.format(np.corrcoef(data[:, 3], data[:, 4])[0, 1]),
                    fontsize=20)

        # set axis
        self.ax1.axis([128.284, 128.295, 37.995, 38.006])
        self.ax1.set_xlabel('PosLon')
        self.ax1.set_ylabel('PosLat')
        self.ax1.set_title('Global Trajectory with Preview (5s)')

        self.ax2.axis([-120, 120, -10, 250])  # prev_dist_show = 50
        self.ax2.set_xlabel('PreviewLon')
        self.ax2.set_ylabel('PreviewLat')
        self.ax2.set_title('Preview Trajectory (5s)')

        self.ax3.set_xlabel('Distance')
        self.ax3.set_ylabel('|Curvature|', color='r')
        self.ax3.set_title('Preview Curvature vs. Speed (1min.)')
        # 달려온 기록을 보여주는 것이므로, preview에서가 아니라,
        # 달려왔던 길에서의 curvature랑 speed를 비교해야 할 거 같긴한데...
        # 저 파란색을 예측하는 것이 목표라고 하면, 뭐 괜찮을 거 같기도..함..

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
            lat_preview, lon_preview = self.previewHelper.get_preview_plane(idx)
            dist_preview, k_preview = self.previewHelper.get_preview_curve(idx, medfilt=15)
            carSpeed = self.df['GPS_Speed'].iloc[idx:idx+self.preview_distance].values
            posLat = self.df['PosLat'].iloc[self.startIdx:idx+self.prev_dist_show].values
            posLon = self.df['PosLon'].iloc[self.startIdx:idx+self.prev_dist_show].values

            yield np.c_[lat_preview, lon_preview, dist_preview, abs(k_preview),
                        (-1)*carSpeed], posLat, posLon  # (preview_distance, 5)

    def update(self, i):
        data, lat, lon = self.stream.__next__()

        self.scat.set_offsets(np.c_[lon[:-self.prev_dist_show], lat[:-self.prev_dist_show]])
        self.scat_prev.set_offsets(np.c_[lon[-self.prev_dist_show:], lat[-self.prev_dist_show:]])

        self.line2[0].set_data(data[:self.prev_dist_show, 1],
                               data[:self.prev_dist_show, 0])

        self.line3.set_data(data[:, 2], data[:, 3])
        # 여기서 확인해야 할 점은, 누적 거리-곡률 그래프에, 속력을 그대로 입혀도,
        # 같은 시점에서 수집된 데이터 맞겠지? 하는 것.

        self.line4.set_data(data[:, 2], data[:, 4])

        self.text.set_text('corr: {}'.format(np.corrcoef(data[:, 3], data[:, 4])[0, 1]))

        for ax in [self.ax1, self.ax2, self.ax3, self.ax3a]:
            ax.grid()

        print(i)

        return self.scat, self.line2[0], self.line3, self.line4, self.text,

class AnimatedScatter:
    def __init__(self, df, startIdx=0, finalIdx=None):
        self.df = df
        self.startIdx = startIdx

        if finalIdx is None:
            self.finalIdx = len(df)
        else:
            self.finalIdx = finalIdx

        assert sum([0 if x in df.columns else 1 for x in
            ['PosLat', 'PosLon']]) == 0, 'Data should contain "PosLat" and "PosLon" fields'

        self.stream = self.data_stream()

        self.fig, self.ax = plt.subplots()

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                        init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        data = self.stream.__next__()
        self.scat = self.ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        self.scat2 = self.ax.scatter(data[-1:, 0], data[-1:, 1], color='r')
        xmax = np.max(self.df['PosLon'])
        xmin = np.min(self.df['PosLon'])
        xstd = np.std(self.df['PosLon'])
        ymax = np.max(self.df['PosLat'])
        ymin = np.min(self.df['PosLat'])
        ystd = np.std(self.df['PosLat'])
        self.ax.axis([xmin-xstd, xmax+xstd, ymin-ystd, ymax+ystd])
        self.ax.set_xlabel('PosLon')
        self.ax.set_ylabel('PosLat')
        self.ax.set_title('Global Trajectory')
        return self.scat, self.scat2 # iterable

    def data_stream(self):
        for i in range(self.startIdx, self.finalIdx):
            lat = self.df['PosLat'].iloc[self.startIdx:i].values
            lon = self.df['PosLon'].iloc[self.startIdx:i].values
            print('{}/{}'.format(i, self.finalIdx), end='\r')
            yield np.c_[lon, lat]  # (# rows, 2)
        print('{}/{}'.format(i, self.finalIdx))

    def update(self, i):
        data = self.stream.__next__()
        self.scat.set_offsets(data[:, :2])
        self.scat2.set_offsets(data[-1:, :2])
        return self.scat, self.scat2

class AnimatedSubplots:
    def __init__(self, df, startIdx=0, finalIdx=None,
                preview_distance=600, preview_time=10, previewType='DISTANCE'):
        self.df = df
        self.startIdx = startIdx
        self.finalIdx = finalIdx

        if finalIdx is None:
            self.finalIdx = len(df)
        else:
            self.finalIdx = finalIdx

        assert sum([0 if x in df.columns else 1 for x in
        ['TimeStamp', 'PosLat', 'PosLon', 'PosLocalX', 'PosLocalY', 'GPS_Speed', 'AngleTrack']]) == 0, 'DataFrame Format MisMatch'

        assert previewType == 'DISTANCE' or previewType == 'TIME', 'Valid previewType: "DISTANCE" or "TIME"'

        self.previewType = previewType

        self.previewHelper = DataHelper(self.df)

        ''' Distance '''
        self.preview_distance = preview_distance
        self.previewHelper.set_preview_distance(preview_distance)

        ''' Time '''
        self.preview_time = preview_time
        self.previewHelper.set_preview_time(preview_time)

        self.stream = self.data_stream()

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 8))

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                        init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        lat, lon, dist, k, speed = self.stream.__next__()
        # data = self.stream.__next__()

        '''
        lat_preview:    data[:, 0]
        lon_preview:    data[:, 1]
        dist_preview:   data[:, 2]
        abs(k_preview): data[:, 3]
        (-1)*carSpeed:  data[:, 4]
        '''

        self.line1 = self.ax1.plot(lon, lat)
        self.line2, = self.ax2.plot(dist, k, color='r')  # curvature

        self.ax2a = self.ax2.twinx()
        self.line3, = self.ax2a.plot(dist, speed, color='b')  # something

        # set axis
        self.ax1.set_xlim(np.min(lon)-0.001, np.min(lon+0.001))
        self.ax1.set_ylim(np.min(lat)-0.001, np.min(lat+0.001))

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
            lat_preview = preview['PosLocalY']
            lon_preview = preview['PosLocalX']
            dist_preview = preview['Distance']
            k_preview = preview['Curvature']
            carSpeed = preview['GPS_Speed']
            print('{}/{}'.format(i, self.finalIdx), end='\r')
            # lat_preview, lon_preview = self.previewHelper.get_preview_plane(idx)
            # dist_preview, k_preview = self.previewHelper.get_preview_curve(idx)
            # carSpeed = self.df['GPS_Speed'].iloc[idx:idx+self.preview_distance].values
            yield lat_preview, lon_preview, dist_preview, abs(k_preview), (-1)*carSpeed
            # yield np.c_[lat_preview, lon_preview, dist_preview, abs(k_preview), (-1)*carSpeed]  # (preview_distance, 5)
        print('{}/{}'.format(i, self.finalIdx))

    def update(self, i):
        lat, lon, dist, k, speed = self.stream.__next__()

        self.line1[0].set_data(lon, lat)
        self.line2.set_data(dist, k)  # 여기서 확인해야 할 점은, 누적 거리-곡률 그래프에, 속력을 그대로 입혀도,
        self.line3.set_data(dist, speed)  # 같은 시점에서 수집된 데이터 맞겠지? 하는 것.

        self.ax1.set_xlim(np.min(lon)-0.001, np.max(lon+0.001))
        self.ax1.set_ylim(np.min(lat)-0.001, np.max(lat+0.001))

        for ax in [self.ax1, self.ax2, self.ax2a]:
            ax.grid()

        print(i)

        return self.line1[0], self.line2, self.line3,

if __name__ == "__main__":
    ''' GraphUtils Example Code
        (datafile: std_xxx.csv (Record in time order),
                   pos_xxx.csv (Record in location order)) '''

    df = pd.read_csv('std_001.csv')
    # df = pd.read_csv('pos_001.csv')

    ''' 주행 궤적 기록 애니메이션 '''
    # ani = AnimatedScatter(df)
    # plt.show()

    ''' 주행 preview & curvature-GPS_SPeed 기록 애니메이션 '''
    ani = AnimatedSubplots(df, previewType='DISTANCE')
    plt.show()

    ''' 주행 궤적 기록 + 주행 preview & Curvature-Speed 기록 애니메이션 '''
    # ani = AnimatedPlots(df, startIdx=270, finalIdx=1810)
    # plt.show()
