import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
from test_data_mapV import *
from test_model import *
import math
import time

previewDistance = 250  # [250m]
histLen = 1  # [1s]
targetLen = 20  # [2s]

def getStationIdx(curPosition, testEnv):
    eDist = np.array([np.sqrt(sum((xy - curPosition)**2)) for xy in testEnv.map_center_xy])
    cur_idx_candi = np.where(eDist == eDist.min())
    cur_idx = cur_idx_candi[0][0]
    return cur_idx

def getCurHeading(curPosition, testEnv):
    eDist = np.array([np.sqrt(sum((xy - curPosition)**2)) for xy in testEnv.map_center_xy])
    cur_idx_candi = np.where(eDist == eDist.min())
    cur_idx = cur_idx_candi[0][0]
    return testEnv.map_center_heading[cur_idx]

def getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed, testEnv):
    distance = (curSpeed_prev + 0.5*(curSpeed-curSpeed_prev))/36
    nextPosition = [curPosition[0] + distance*np.sin(curHeading),
                    curPosition[1] + distance*np.cos(curHeading)]  # root?
    eDist = np.array([np.sqrt(sum((xy - nextPosition)**2)) for xy in testEnv.map_center_xy[curIdx+1:curIdx+int(np.ceil(distance))+10]])
    next_idx_candi = np.where(eDist == eDist.min())
    next_idx = next_idx_candi[0][0] + (curIdx+1)  # 0 -> curIdx+1이므로,
    return next_idx, nextPosition

def getNewPositionWithOffset(nextPosition, curHeading, lateralOffset):  # 부호 방향 수정 필요
    if lateralOffset > 0:
        heading = curHeading + math.radians(90)  # offset이 +일 때, center line의 오른쪽에 있도록 계산하였음. 그리고 +y, +x, -y, -x 순으로 0, 90, 180, 270인듯 함 (python에서)
    else:
        heading = curHeading - math.radians(90)

    distance = abs(lateralOffset)
    nextPosition_wOffset = [nextPosition[0] + distance*np.sin(heading),
                            nextPosition[1] + distance*np.cos(heading)]
    return nextPosition_wOffset


def mimic(testModel, mapFile, testEnv, initialSpeed):
    globalXs = []  # output
    globalYs = []
    globalSs = []

    histSpeeds_simul = [initialSpeed] * histLen * 10
    curSpeed = initialSpeed

    curAccelX = 0.0  # 안쓰임.
    histAccelX_simul = [curAccelX] * histLen * 10  # 안쓰임.

    curOffset = 0.0
    histOffset_simul = [curOffset] * histLen * 10


    df_map = pd.read_csv(mapFile)
    testDataStream = testEnv.get_preview()

    total_preds = []
    total_positions = []

    stationIdxs = []

    sTime = time.time()
    curIdx = 0
    while curIdx < len(df_map):
        curPosition = testEnv.curPosition

        if curIdx == 0:
            total_positions.append(curPosition)

        mapPreview, transMapPreview, curvature = testDataStream.__next__()

        aPreds, dlPreds, alphas = testModel.predict(curvature[1:1+previewDistance], curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul, curOffset, histOffset_simul)

        vPreds = np.cumsum(aPreds) + curSpeed
        lPreds = np.cumsum(dlPreds) + curOffset

        # predictions = np.vstack((vPreds, aPreds)).T
        predictions = np.vstack((vPreds, aPreds, lPreds)).T

        total_preds.append(predictions)
        stationIdxs.append(getStationIdx(curPosition, testEnv))

        print('(MapLength:{}/{})\tLapTime: {:.4f}s\tSpeed: {:.4f}\tOffsets: {:.4f}'.format(curIdx, len(df_map), np.array(total_preds).shape[0]*0.1, vPreds[0], lPreds[0]), end='\r')

        globalXs.append(curPosition[0])
        globalYs.append(curPosition[1])
        globalSs.append(curSpeed)

        ''' Speed update '''
        curSpeed_prev = curSpeed
        curAccelX_prev = curAccelX  # 안쓰임.

        curSpeed = predictions[0][0]
        histSpeeds_simul.pop(0)
        histSpeeds_simul += [curSpeed_prev]

        ''' Lateral Offset update '''
        curOffset_prev = curOffset
        curOffset = predictions[0][2]
        histOffset_simul.pop(0)
        histOffset_simul += [curOffset_prev]

        ''' Location update '''
        curHeading = getCurHeading(curPosition, testEnv)
        curIdx, nextPosition = getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed, testEnv)
        nextPosition_wOffset = getNewPositionWithOffset(nextPosition, curHeading, lPreds[0])

        total_positions.append(nextPosition_wOffset)

        testEnv.curIdx = curIdx
        testEnv.curPosition = nextPosition

    print('(MapLength:{}/{})\tLapTime: {:.4f}s\tSpeed: {:.4f} ({}s)\tOffsets: {:.4f}'.format(curIdx, len(df_map), np.array(total_preds).shape[0]*0.1, time.time()-sTime, vPreds[0], lPreds[0]))
    return np.array(globalXs), np.array(globalYs), np.array(globalSs)


if __name__ == "__main__":
    ''' Env Parameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', '-d')
    parser.add_argument('--vNumber', '-v')
    parser.add_argument('--epoch', '-e')
    parser.add_argument('--circuit', '-c')
    parser.add_argument('--initialSpeed', '-s')

    args = parser.parse_args()

    driver = args.driver
    vNumber = args.vNumber
    epoch = args.epoch
    circuit = args.circuit
    initialSpeed = args.initialSpeed

    ''' Driver Model '''
    chptFolderPath = './chpt_{}_v{}'.format(driver, vNumber)  # chpt folder path
    chpt_enc_path = '{}/checkpoint_ENC_{}.pth.tar'.format(chptFolderPath, epoch)
    chpt_dec_path = '{}/checkpoint_DEC_{}.pth.tar'.format(chptFolderPath, epoch)
    chpt_stat_path = '{}/stat_{}.pickle'.format(chptFolderPath, epoch)

    testModel = Model(chpt_enc_path, chpt_dec_path, chpt_stat_path)

    ''' Map '''
    mapFile = '../Data/mapData/{}_norm.csv'.format(circuit)
    testEnv = TestEnv(cWindow=previewDistance, vpWindow=histLen, mapfile=mapFile, repeat=1)

    gx, gy, gs = mimic(testModel, mapFile, testEnv, float(initialSpeed))

    plt.figure()
    try:
        df_map = pd.read_csv('../Data/mapData/{}.csv'.format(circuit))
        plt.plot(df_map['inner_x'], df_map['inner_y'], 'k--')
        plt.plot(df_map['outer_x'], df_map['outer_y'], 'k--')
    except:
        pass
    plt.plot(gx, gy, 'r--', label='Model')

    plt.figure()
    plt.plot(gs)
    plt.xlabel('Time [0.1s]')

    plt.legend()
    plt.show()

    df = pd.DataFrame()
    df['PosLocalX'] = gx
    df['PosLocalY'] = gy
    df['GPS_Speed'] = gs

    df.to_csv('results_{}.csv'.format(circuit))
    print('SAVED:: results_{}.csv'.format(circuit))
