import pdb
from test_data_mapV import *
from test_model import *

import numpy as np
import matplotlib.pyplot as plt

import time
import argparse
import scipy
import os
from tqdm import tqdm
import time

''' ex. python simulation.py -d YJ -v 1 -e 30 -c mugello '''

''' Setting Env. Parameters '''
previewDistance = 250
historyLength = 1  # unit [sec.], 차량의 현재 state를 정의할 때 사용되는 직전 과거 속도 프로파일

initialSpeed = 0
histSpeeds_simul = [initialSpeed] * historyLength * 10
curSpeed = initialSpeed

curAccelX = 0.0  # Not used.
histAccelX_simul = [curAccelX] * historyLength * 10  # Not used.

parser = argparse.ArgumentParser()
parser.add_argument('--driver', '-d')
parser.add_argument('--vNumber', '-v')
parser.add_argument('--epoch', '-e')
parser.add_argument('--circuit', '-c')
# parser.add_argument('--mode', '-m')  # m 1 (simulation mode)

args = parser.parse_args()

driver = args.driver
vNumber = args.vNumber
epoch = args.epoch
circuit = args.circuit

''' Load Driver Model '''
chptFolderPath = './chpt_{}_v{}'.format(driver, vNumber)  # chpt folder path
chpt_enc_path = '{}/checkpoint_ENC_{}.pth.tar'.format(chptFolderPath, epoch)
chpt_dec_path = '{}/checkpoint_DEC_{}.pth.tar'.format(chptFolderPath, epoch)
chpt_stat_path = '{}/stat_{}.pickle'.format(chptFolderPath, epoch)

testModel = Model(chpt_enc_path, chpt_dec_path, chpt_stat_path)  # speed prediction에 사용할 model parameters

''' Load Circuit Map '''
repeat = 3
mapfile = '../Data/mapData/{}_norm.csv'.format(circuit)
testEnv = TestEnv(cWindow=previewDistance, vpWindow=historyLength, mapfile=mapfile, repeat=repeat)
testDataStream = testEnv.get_preview()

resultPath = 'result_mapV_{}_{}_v{}_{}_{}kph'.format(driver, circuit, vNumber, epoch, initialSpeed)
print('Check the result @ ', resultPath)
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

def getCurHeading(curPosition):
    # circuit의 center line 중 현재 위치와 가장 가까운 점의 heading[rad.]을 가져옴.
    eDist = np.array([np.sqrt(sum((xy - curPosition)**2)) for xy in testEnv.map_center_xy])
    cur_idx_candi = np.where(eDist == eDist.min())
    cur_idx = cur_idx_candi[0][0]
    return testEnv.map_center_heading[cur_idx]

def getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed):
    # 현재 지점에서 현재 지점의 heading 추정 값(road centerline을 이용)과,
    # 현재 지점의 속도와 0.1초 후의 예측된 속도를 이용하여,
    # 현재 지점에서 heading으로 0.1초동안 이동했을 때의 위치값을 계산
    distance = (curSpeed_prev + 0.5*(curSpeed-curSpeed_prev))/36
    nextPosition = [curPosition[0] + distance*np.sin(curHeading),
                    curPosition[1] + distance*np.cos(curHeading)]  # 여기 root하는게 맞지 않나?

    eDist = np.array([np.sqrt(sum((xy - nextPosition)**2)) for xy in testEnv.map_center_xy[curIdx+1:curIdx+int(np.ceil(distance))+10]])
    next_idx_candi = np.where(eDist == eDist.min())  # curIdx 근처에서 고르도록 하여, idx가 작아지는 것을 방지.
    next_idx = next_idx_candi[0][0] + (curIdx+1)  # 0 -> curIdx+1이므로,

    return next_idx, nextPosition

def getCenterLinePosition(curPosition):
    eDist = np.array([np.sqrt(sum((xy - curPosition)**2)) for xy in testEnv.map_center_xy])
    cur_idx_candi = np.where(eDist == eDist.min())
    cur_idx = cur_idx_candi[0][0]
    return cur_idx, testEnv.map_center_xy[cur_idx]  # for carSim

def main():

    global curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul  # accel은 사용 안함

    total_preds = []
    total_alphas = []

    data4CarSim = []  # carSim에서는 centerline 위 속도 프로파일이 필요하다 함. [[x,y,stationIdx,speed],]
    curIdx = 0

    while curIdx < len(testEnv.df_map)*repeat:
        print('LapTime: {:.4f}s ({:5d}/{})'.format(np.array(total_preds).shape[0]*0.1, curIdx, len(testEnv.df_map)*repeat), end='\r')

        curPosition = testEnv.curPosition  # 현재 지점 reload

        # for carSim
        stationIdx, curPositionAtCenterLine = getCenterLinePosition(curPosition)
        data4CarSim.append([curPositionAtCenterLine[0], curPositionAtCenterLine[1], stationIdx, curSpeed])

        try:  # preview 계산
            mapPreview, transMapPreview, curvature = testDataStream.__next__()  # [currentPosition] + [preview (250,)]
        except:
            break

        # speed prediction
        # curTime = time.time()
        aPreds, alphas = testModel.predict(curvature[1:1+previewDistance], curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul)
        # print(time.time() - curTime)
        # pdb.set_trace()

        vPreds = np.cumsum(aPreds) + curSpeed
        predictions = np.vstack((vPreds, aPreds)).T

        total_preds.append(predictions)
        total_alphas.append(np.repeat(np.mean(alphas, axis=0), 2, axis=0))

        # speed update
        curSpeed_prev = curSpeed

        curSpeed = predictions[0][0]
        histSpeeds_simul.pop(0)
        histSpeeds_simul += [curSpeed_prev]

        # location update
        curHeading = getCurHeading(curPosition)
        curIdx, nextPosition = getNewPosition(curIdx, curPosition, curHeading, curSpeed_prev, curSpeed)
        testEnv.curIdx = curIdx
        testEnv.curPosition = nextPosition

        # location update visualization
        # if curIdx > 200:
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(111)
        #
        #     ax1.plot(testEnv.map_center_xy[:200, 0], testEnv.map_center_xy[:200, 1], 'k')
        #     ax1.scatter(mapPreview[:, 0], mapPreview[:, 1], color='r', s=5)  # [0]: almost current position
        #     ax1.plot(curPosition[0], curPosition[1], 'b.')
        #     ax1.plot(nextPosition[0], nextPosition[1], 'bx')
        #     ax1.plot([curPosition[0], nextPosition[0]], [curPosition[1], nextPosition[1]], 'b')
        #     plt.show()

    print('Simulation END\t\t\t\t\t\t\t\t\t\t')
    print('LapTime: {:.4f}s ({:5d}/{})'.format(np.array(total_preds).shape[0]*0.1, curIdx, len(testEnv.df_map)*repeat))

    np.save('{}/total_preds'.format(resultPath), np.array(total_preds))
    np.save('{}/total_alphas'.format(resultPath), np.array(total_alphas))
    np.save('{}/data4CarSim'.format(resultPath), np.array(data4CarSim))

    data4CarSim = np.array(data4CarSim)
    idx_candi = np.where(data4CarSim[:, -2]<10)[0]
    print(idx_candi)
    a = int(input('input a'))
    b = int(input('input b'))
    for i in range(repeat):  # 3만 지원.
        if i == 0:
            df = pd.DataFrame(data4CarSim[:a], columns=['center_x', 'center_y', 'stationIdx', 'speed'])
        elif i == 1:
            df = pd.DataFrame(data4CarSim[a:b], columns=['center_x', 'center_y', 'stationIdx', 'speed'])
        elif i == 2:
            df = pd.DataFrame(data4CarSim[b:], columns=['center_x', 'center_y', 'stationIdx', 'speed'])
        df.to_csv('{}/data4CarSim_{}_{}.csv'.format(resultPath, driver, i))

    for j in range(len(total_preds)):
        plt.plot(range(j, j+20), total_preds[j][:, 0], 'k.-', alpha=0.1)
    plt.plot(np.array(total_preds)[:, 0, 0], 'r--')  # 0: 예측한 시각 index // 1: 매 예측에서 예측값의 index, [0]: ^v_(t+1) // 2: [0]: 속도, [1]: 가속도
    plt.show()


if __name__ == "__main__":
    main()
