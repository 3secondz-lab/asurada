import pdb
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper
from helper import *
from utils import *

class TestEnv:
    def __init__(self, datafile, recFreq, cWindow=10.0, vpWindow=2, cUnit=10, vpUnit=10, condition=None, st=None, et=None, mapfile=None):
        self.df = pd.read_csv(datafile)
        self.recFreq = recFreq
        if 'Speed' in self.df.columns and 'GPS_Speed' not in self.df.columns:
            self.df = self.df.rename(columns={'Speed':'GPS_Speed'})

        if mapfile is not None:
            # self.mapfile = mapfile
            self.df_map, self.map_center_xy = replicateMapInfo(mapfile, self.df)

        self.dh = DataHelper(self.df)  # for calculating preview curvature without mapfile
        # self.dh.set_preview_time(cWindow)  # unit [s]  # 이렇게 썼었는데, 사실상 preview는 거리 base가 되어야 한다고 생각함!

        # self.cWindow_d = 100.0
        # self.dh.set_preview_distance(self.cWindow_d)  # 임의로 해봄..
        self.dh.set_preview_distance(cWindow)  # unit [m]

        self.cWindow = cWindow
        self.vpWindow = vpWindow

        self.condition = condition
        if condition is not None:
            # self.st = 861  # yy/
            # self.et = 955

            # self.st = 13137  # ij/set2-2 (valid)
            # self.et = 13341

            # self.st = 22174  # ij/set4-2 (test)
            # self.et = 22376

            self.st = st
            self.et = et

    def get_preview(self):
        for idx in range(len(self.df)):

            if self.condition is not None:
                timeStamp = self.df['TimeStamp'].iloc[idx]
                if timeStamp < self.st:
                    continue
                if timeStamp > self.et:
                    continue

            preview = self.dh.get_preview(idx, method = 'DISTANCE')  # currentSpeed 때문에 df_map 사용과 상관없이 필요함

            if hasattr(self, 'df_map'):
                ss, kk = getMapPreview(idx, self.df, self.df_map, self.map_center_xy, self.cWindow, self.dh)
                curvature = kk
            else:
                # preview = self.dh.get_preview(idx, method='TIME')
                # preview = self.dh.get_preview(idx, method = 'DISTANCE2')
                # previewX = preview['PreviewX']
                # previewY = preview['PreviewY']
                curvature = preview['Curvature']

            currentSpeed = preview['GPS_Speed'][0]
            # currentThrottle = preview['ECU_THROTTLE'][0]
            # currentSteer = preview['ECU_STEER_SPD'][0]

            # try:
            # histSpeeds = self.df['GPS_Speed'].values[idx-int(self.recFreq*self.vpWindow):idx].tolist()
            histSpeeds = self.df['GPS_Speed'].values[idx-int(self.recFreq*self.vpWindow):idx]
            # except:
                # histSpeeds = self.df['speed'].values[idx-int(self.recFreq*self.vpWindow):idx].tolist()

            # if len(previewX)-1 < self.recFreq * self.previewT:  # not long enough preview
            # if len(curvature) < self.recFreq * self.cWindow:  # not long enough preview
            #     continue  # 여기 임시로 없앰

            if len(histSpeeds) < self.recFreq * self.vpWindow:
                continue

            # yield preview
            # yield curvature, currentSpeed, currentThrottle, currentSteer, histSpeeds
            # pdb.set_trace()
            yield curvature, currentSpeed, histSpeeds
            # yield previewX, previewY, curvature, currentSpeed, currentThrottle, currentSteer
