import pdb
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper
# from helper import *
from utils import *

def getMapPreview_mapV(i, cur_xy, df_map, map_center_xy, previewDistance, transform=False, headingInfo=None):
    eDist = np.array([np.sqrt(sum((xy - cur_xy)**2)) for xy in map_center_xy])

    try:
        cur_idx_candi = np.where(eDist == eDist.min())
        temp = (cur_idx_candi[0] - i)
        cur_idx = cur_idx_candi[0][np.where(temp>=0)[0][0]]
        cur_dist = df_map['distance'][cur_idx]  # map의 시작을 0으로 했을 때, cur_xy까지의 거리
        preview_end_idx = np.argmin(abs(df_map['distance'].values - (cur_dist + previewDistance)))
    except:
        return [], [], [], []

    map_preview = map_center_xy[cur_idx:preview_end_idx+1, :]  # curvature[0]을 현재 위치로. for visualization
    dist, curvature = station_curvature(map_preview[:, 0], map_preview[:, 1], medfilt=15)

    if transform:
        transX, transY = transform_plane(map_preview[:, 0]-map_preview[0, 0],
                                            map_preview[:, 1]-map_preview[0, 1], headingInfo[cur_idx])
        return dist, curvature, map_preview, np.column_stack((transX, transY))

    else:
        return dist, curvature, map_preview

class TestEnv:
    def __init__(self, cWindow=250.0, vpWindow=1, vpUnit=10, mapfile=None, repeat=None):

        # datafile, recFreq,

        # df가 없음. 초기 속도 벡터만 만들어주면 됨. constant 벡터로.

        self.repeat = repeat

        self.df_map = pd.read_csv(mapfile)
        self.df_map_replicated = pd.concat([self.df_map]*(self.repeat+1), ignore_index=True)
        self.df_map_replicated['distance'] = [*range(len(self.df_map)*(self.repeat+1))]
        self.map_center_xy = self.df_map_replicated[['center_x', 'center_y']].values

        self.map_center_heading = np.mod(np.pi/2 - np.arctan2(np.gradient(self.map_center_xy[:, 1]),
                                                              np.gradient(self.map_center_xy[:, 0])), 2*np.pi)

        # map_center_xy와 map_center_heading의 해석
        # --> simulation을 할 때, 매 curPosition에서  curPosition + sqrt(d)*np.sin(self.map_center_heading), curPosition + sqrt(d)*np.cos(self.map_center_heading)
        # --> 로 하면 될듯함.
        # plt.figure()
        # plt.plot(self.map_center_xy[200:250, 0], self.map_center_xy[200:250:, 1], 'k.--')
        # plt.plot(self.map_center_xy[249, 0], self.map_center_xy[249, 1], 'bx')
        # for i in range(200, 250, 5):
        #     xx = self.map_center_xy[i][0]
        #     yy = self.map_center_xy[i][1]
        #     xxx = np.sin(self.map_center_heading[i])
        #     yyy = np.cos(self.map_center_heading[i])
        #     plt.plot([xx, xx+xxx], [yy, yy+yyy], 'r-')
        #     plt.plot([xx], [yy], 'ro')
        #     plt.plot([xx+xxx], [yy+yyy], 'rx')
        # plt.show()

        self.previewDistance = cWindow

        self.histLen = vpWindow * vpUnit

        self.targetLen = 2 * vpUnit

        self.curIdx = 0  # class 밖에서 update.
        self.curPosition = [self.df_map.iloc[0]['center_x'], self.df_map.iloc[0]['center_y']]  # 밖에서 update.

    def get_preview(self):
        idx = self.curIdx
        while idx < len(self.df_map)*self.repeat:

            idx = self.curIdx
            curPosition = self.curPosition  # 바깥에서, 이 함수를 부르기 전에 update.
            dist, curvature, mapPreview, transMapPreview = getMapPreview_mapV(idx, curPosition, self.df_map_replicated, self.map_center_xy,
                                                self.previewDistance, transform=True, headingInfo=self.map_center_heading)
            # training에서는 curvature[0]을 1m ahead로 놨는데, 여기서는 그림을 그려야 하니까, [0]을 0m (현재 위치)로 놓고,
            # prediction할때만 curvature[1:]로

            yield mapPreview, transMapPreview, curvature
            # , curSpeed, histSpeed, curAccelX, histAccelX, targetSpeeds, targetAccelXs
