import pdb
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper
from helper import *
from utils import *

from test_constants import *

class TestEnv:
    def __init__(self, datafile, recFreq, cWindow=250.0, vpWindow=2, cUnit=10, vpUnit=10, condition=None, st=None, et=None, mapfile=None):

        self.df = pd.read_csv(datafile)
        if 'Speed' in self.df.columns and 'GPS_Speed' not in self.df.columns:
            self.df = self.df.rename(columns={'Speed':'GPS_Speed'})

        self.recFreq = recFreq
        self.previewDistance = cWindow

        if mapfile is not None:
            self.df_map = pd.read_csv(mapfile)
            self.df_map_replicated, self.map_center_xy = replicateMapInfo(self.df_map, self.df, vis=False)
            self.map_center_heading = np.mod(np.pi/2 - np.arctan2(np.gradient(self.map_center_xy[:, 1]),
                                                                  np.gradient(self.map_center_xy[:, 0])), 2*np.pi)

        self.histLen = vpWindow * vpUnit

        self.targetLen = predLength

        self.condition = condition
        if condition is not None:
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

            curPosition = [self.df.iloc[idx]['PosLocalX'], self.df.iloc[idx]['PosLocalY']]
            dist, curvature, mapPreview, transMapPreview = getMapPreview(idx, self.df, self.df_map_replicated, self.map_center_xy,
                                                self.previewDistance, transform=True, headingInfo=self.map_center_heading)
            # training에서는 curvature[0]을 1m ahead로 놨는데, 여기서는 그림을 그려야 하니까, [0]을 0m (현재 위치)로 놓고,
            # yield할때는 curvature[1:]로 해야하나? -> prediction할때만 curvature[1:]로 하면 됨.

            if idx < self.histLen:
                histSpeed = []
                histAccelX = []
                # histAccelY = []
            else:
                histSpeed = self.df['GPS_Speed'].iloc[idx-self.histLen:idx].values
                histAccelX = self.df['AccelX'].iloc[idx-self.histLen:idx].values
                # histAccelY = self.df['AccelY'].iloc[idx-self.histLen:idx].values
            curSpeed = self.df['GPS_Speed'].iloc[idx]
            curAccelX = self.df['AccelX'].iloc[idx]
            # curAccelY = self.df['AccelY'].iloc[idx]

            targetSpeeds = self.df['GPS_Speed'].iloc[idx+1:idx+self.targetLen+1].values
            targetAccelXs = self.df['AccelX'].iloc[idx+1:idx+self.targetLen+1].values
            # accelY = self.df['AccelY'].iloc[idx+1:idx+self.targetLen]

            if len(histSpeed) > 0:
                try:
                    temp = np.array(histSpeed.tolist() + [curSpeed] + targetSpeeds.tolist())
                except:
                    pdb.set_trace()
                tempDiff = np.diff(temp)
                tempDiff = np.array(tempDiff.tolist() + [tempDiff[-1]])

                histAccelX = tempDiff[:self.histLen]
                curAccelX = tempDiff[self.histLen]
                # targetAccelXs = tempDiff[-self.targetLen:]

                targetAccelXs = tempDiff[-self.targetLen-1:]  # v_t까지 알고 있는 상황에서 a_t부터 예측해야 하므로, 이게 맞음.

            yield curPosition, mapPreview, transMapPreview, dist, curvature, curSpeed, histSpeed, curAccelX, histAccelX, targetSpeeds, targetAccelXs
