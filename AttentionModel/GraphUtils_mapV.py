import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import argparse
from test_data_mapV import *
from test_model import *

previewDistance = 250
histLen = 1
targetLen = 20

initialSpeed = 0
histSpeeds_simul = [initialSpeed] * histLen * 10
curSpeed = initialSpeed

curAccelX = 0.0  # 안쓰임.
histAccelX_simul = [curAccelX] * histLen * 10  # 안쓰임.

def getCurHeading(curPosition):
    eDist = np.array([np.sqrt(sum((xy - curPosition)**2)) for xy in testEnv.map_center_xy])
    cur_idx_candi = np.where(eDist == eDist.min())
    cur_idx = cur_idx_candi[0][0]
    return testEnv.map_center_heading[cur_idx]

def getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed):
    distance = (curSpeed_prev + 0.5*(curSpeed-curSpeed_prev))/36
    nextPosition = [curPosition[0] + distance*np.sin(curHeading),
                    curPosition[1] + distance*np.cos(curHeading)]  # root?
    eDist = np.array([np.sqrt(sum((xy - nextPosition)**2)) for xy in testEnv.map_center_xy[curIdx+1:curIdx+int(np.ceil(distance))+10]])
    next_idx_candi = np.where(eDist == eDist.min())
    next_idx = next_idx_candi[0][0] + (curIdx+1)  # 0 -> curIdx+1이므로,
    return next_idx, nextPosition

class AnimatedRecords:
    def __init__(self, testModel, mapFile, testEnv):
        self.df_map = pd.read_csv(mapFile)
        self.map_center_x = self.df_map['center_x']
        self.map_center_y = self.df_map['center_y']

        self.testEnv = testEnv
        self.testDataStream = self.testEnv.get_preview()
        self.stream = self.data_stream()

        self.fig = plt.figure(figsize=(12, 10))
        self.ax1 = self.fig.add_subplot(221)  # circuit + curPosition + preview-track
        self.ax2 = self.fig.add_subplot(222)  # preview-track shape (0, 0)
        self.ax3 = self.fig.add_subplot(212)  # predicted speed

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=2,
                                        init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        global previewDistance, targetLen

        curPosition, mapPreview, transMapPreview, curvature, curSpeed, vPreds, total_preds = self.stream.__next__()

        self.lineCenter = self.ax1.plot(self.map_center_x, self.map_center_y, 'k')

        self.curPosition = self.ax1.scatter(curPosition[0], curPosition[1], color='b')
        self.mapPreview = self.ax1.scatter(mapPreview[:, 0], mapPreview[:, 1], color='r', s=5)

        self.mapPreviewTrans = self.ax2.scatter(transMapPreview[:, 1], transMapPreview[:, 0])
        self.ax2.set_ylim(0, previewDistance)
        self.ax2.set_xlim(-previewDistance, previewDistance)

        self.predSpeeds, = self.ax3.plot(range(len(total_preds)), total_preds[:, 0, 0], 'b.--')  # 매 point에서 현재 속력만 기록
        self.ax3.set_xlim(-1, 2000)
        self.ax3.set_ylim(-1, 200)

        return self.lineCenter, self.curPosition, self.mapPreview, self.mapPreviewTrans, self.predSpeeds, #self.curSpeeds, self.predictions,

    def data_stream(self):
        global curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul

        total_preds = []
        curIdx = 0
        while curIdx < len(self.df_map):
            # print('{:5d}/{}'.format(curIdx, len(self.df_map)), end='\r')

            curPosition = self.testEnv.curPosition

            mapPreview, transMapPreview, curvature = self.testDataStream.__next__()

            aPreds, alphas = testModel.predict(curvature[1:1+previewDistance], curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul)
            vPreds = np.cumsum(aPreds) + curSpeed
            predictions = np.vstack((vPreds, aPreds)).T

            total_preds.append(predictions)
            print('LapTime: {:.4f}s'.format(np.array(total_preds).shape[0]*0.1), end='\r')
            yield curPosition, mapPreview, transMapPreview, curvature, curSpeed, vPreds, np.array(total_preds),

            ''' Speed update '''
            curSpeed_prev = curSpeed
            curAccelX_prev = curAccelX  # 안쓰임.

            curSpeed = predictions[0][0]
            histSpeeds_simul.pop(0)
            histSpeeds_simul += [curSpeed_prev]

            ''' Location update '''
            curHeading = getCurHeading(curPosition)
            curIdx, nextPosition = getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed)
            self.testEnv.curIdx = curIdx
            self.testEnv.curPosition = nextPosition

    def update(self, i):
        curPosition, mapPreview, transMapPreview, curvature, curSpeed, vPreds, total_preds = self.stream.__next__()

        self.curPosition.set_offsets([curPosition[0], curPosition[1]])
        self.mapPreview.set_offsets(np.c_[mapPreview[:, 0], mapPreview[:, 1]])
        self.mapPreviewTrans.set_offsets(np.c_[transMapPreview[:, 1], transMapPreview[:, 0]])

        self.predSpeeds.set_data(range(len(total_preds)), total_preds[:, 0, 0])

        return self.curPosition, self.mapPreview, self.mapPreviewTrans, self.predSpeeds,


if __name__ == "__main__":
    ''' Env Parameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', '-d')
    parser.add_argument('--vNumber', '-v')
    parser.add_argument('--epoch', '-e')
    parser.add_argument('--circuit', '-c')
    # parser.add_argument('--mode', '-m')  # 얘는 항상 simulation mode

    args = parser.parse_args()

    driver = args.driver
    vNumber = args.vNumber
    epoch = args.epoch
    circuit = args.circuit

    ''' Driver Model '''
    chptFolderPath = './chpt_{}_v{}'.format(driver, vNumber)  # chpt folder path
    resultPath = 'result_mapV_{}_{}_v{}_{}'.format(driver, circuit, vNumber, epoch)
    print(resultPath)
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)

    chpt_enc_path = '{}/checkpoint_ENC_{}.pth.tar'.format(chptFolderPath, epoch)
    chpt_dec_path = '{}/checkpoint_DEC_{}.pth.tar'.format(chptFolderPath, epoch)
    chpt_stat_path = '{}/stat_{}.pickle'.format(chptFolderPath, epoch)

    testModel = Model(chpt_enc_path, chpt_dec_path, chpt_stat_path)  # speed prediction에 사용할 model parameters

    ''' Map '''
    mapFile = '../Data/mapData/{}_norm.csv'.format(circuit)
    testEnv = TestEnv(cWindow=previewDistance, vpWindow=histLen, mapfile=mapFile, repeat=1)

    ani = AnimatedRecords(testModel, mapFile, testEnv)
    plt.show()
