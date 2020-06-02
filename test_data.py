import pdb
import pandas as pd
from helper import *

class TestEnv:
    def __init__(self, datafile, recFreq, previewT=10.0):
        self.df = pd.read_csv(datafile)
        self.dh = DataHelper(self.df)
        self.dh.set_preview_time(previewT)

        self.recFreq = recFreq
        self.previewT = previewT

    def get_preview(self):
        for idx in range(len(self.df)):

            preview = self.dh.get_preview(idx, method='TIME')
            # previewX = preview['PreviewX']
            # previewY = preview['PreviewY']
            curvature = preview['Curvature']
            currentSpeed = preview['GPS_Speed'][0]
            currentThrottle = preview['ECU_THROTTLE'][0]
            currentSteer = preview['ECU_STEER_SPD'][0]

            # if len(previewX)-1 < self.recFreq * self.previewT:  # not long enough preview
            if len(curvature) < self.recFreq * self.previewT:  # not long enough preview
                continue

            # yield preview
            yield curvature, currentSpeed, currentThrottle, currentSteer
            # yield previewX, previewY, curvature, currentSpeed, currentThrottle, currentSteer
