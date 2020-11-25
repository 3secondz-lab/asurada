import pdb
from test_data import *
from test_model import *

import numpy as np
import matplotlib.pyplot as plt

import time
import argparse
import scipy
import os
from tqdm import tqdm

curvatureLength = 250
historyLength = 1  #1  # unit: s
# historyLength = 0  #1  # unit: s  # v31, 32

''' ex. python test_main.py -c imola -v 1 -e 60 -m 1 '''

parser = argparse.ArgumentParser()
parser.add_argument('--driver', '-d')
parser.add_argument('--vNumber', '-v')
parser.add_argument('--epoch', '-e')
parser.add_argument('--circuit', '-c')
parser.add_argument('--mode', '-m')  # 1:simulation mode, 0:check learning result

args = parser.parse_args()

driver = args.driver
vNumber = args.vNumber
# if vNumber == '4':  # test 코드에서는 accel값을 굳이 smoothing을 해야하지 않아도 됨.
#     accelXSmoothing = True  # 예측된 값이 true값의 smoothing된 값으로 되는지만 보면 되고, 그리고 예측 후 과거가 되는 값도 smoothing된 값이 들어가게 되므로,
                              # 결국 training 환경과 같게 됨.
epoch = args.epoch

# LapPoints: calculated by replicateMapInfo (@ utils.py)
lapIdxs_yj = {'mugello': [0, 1656, 3229, 4734, 6244, 7757, 9305, 10868],  # 7
              'magione': [0, 1018, 1961, 2885, 3818, 4742, 5710, 6657, 7617, 8542, 9517],  # 10
              'imola': [0, 1550, 2981, 4429, 5859, 7261, 8679, 10114, 11527],  # 8
              'spa': [0, 2148, 4068, 6014, 7927, 9872, 11815, 13742, 15681, 17681]}  # 9

lapIdxs_ys = {'mugello': [0, 1742, 3851, 5532, 7302, 8973, 10615, 12250, 14019, 15775, 17432, 19140],  # 11
              'magione': [98, 1139, 2163, 3216, 4245, 5262, 6367, 7364, 8374, 9416, 10419],  # 10
              'imola': [0, 1640, 3316, 4939, 6547, 8229, 9756, 11322, 12895, 14491, 14955],
              'spa': [0, 2275, 4436, 6635, 8843, 11101, 13222, 15423, 17564, 19677, 21785]}  # 10, ref를 yj record의 index0으로 잡았을때.

if driver=='YJ':  # test_envset.py
    st=lapIdxs_yj[args.circuit][1]  # 2번째 lap의 data index (원래는 timestamp로 계산했었음)
    et=lapIdxs_yj[args.circuit][2]

elif driver=='YS':
    if args.circuit == 'mugello':  # 두번째 lap은 0에 가까운 구간이 있어서, test에 부적합.
        st=lapIdxs_ys[args.circuit][2]
        et=lapIdxs_ys[args.circuit][3]
    else:
        st=lapIdxs_ys[args.circuit][1]
        et=lapIdxs_ys[args.circuit][2]

datafile = '../Data/driverData/{}/{}-record-scz_msgs.csv'.format(args.driver, args.circuit)
mapfile = '../Data/mapData/{}_norm.csv'.format(args.circuit)
testEnv = TestEnv(datafile,
                 recFreq=10, cWindow=curvatureLength, vpWindow=historyLength, condition=True,
                 st=st, et=et, mapfile=mapfile)
testDataStream = testEnv.get_preview()  # preview를 입력받는 상황을 가정
histLen = testEnv.histLen

chptFolderPath = './chpt_{}_v{}'.format(driver, vNumber)  # chpt folder path
if args.mode=='1':
    resultPath = 'result_{}_{}_v{}_{}'.format(driver, args.circuit, vNumber, epoch)
else:
    resultPath = 'result_{}_{}_v{}_{}_nr'.format(driver, args.circuit, vNumber, epoch)
print(resultPath)
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

chpt_enc_path = '{}/checkpoint_ENC_{}.pth.tar'.format(chptFolderPath, epoch)
chpt_dec_path = '{}/checkpoint_DEC_{}.pth.tar'.format(chptFolderPath, epoch)
chpt_stat_path = '{}/stat_{}.pickle'.format(chptFolderPath, epoch)

testModel = Model(chpt_enc_path, chpt_dec_path, chpt_stat_path)  # speed prediction에 사용할 model parameters

# def draw(idx, mapPreview, curPosition, transMapPreview, curvatures, alphas,
#         curSpeed, realCurrentSpeed, predictions, trueSpeeds, histSpeeds_simul, histSpeeds,
#         curAccelX, realCurrentAccelX, trueAccelXs, histAccelX_simul, histAccelX):
def draw(idx, mapPreview, curPosition, transMapPreview, curvatures, alphas, curSpeed, realCurrentSpeed, predictions, trueSpeeds, histSpeeds_simul, histSpeeds, curAccelX, realCurrentAccelX, trueAccelXs, histAccelX_simul, histAccelX):

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)  # circuit + curPosition + preview-track
    ax2 = fig.add_subplot(222)  # preview-track shape (tranformed)
    ax3 = fig.add_subplot(425)  # speed
    ax5 = fig.add_subplot(427)  # accelX
    ax4 = fig.add_subplot(426)  # curvature, alpha

    ax1.plot(testEnv.map_center_xy[:, 0], testEnv.map_center_xy[:, 1], 'k')
    ax1.scatter(mapPreview[:, 0], mapPreview[:, 1], color='r', s=5)  # [0]: almost current position
    ax1.scatter(curPosition[0], curPosition[1], color='b')

    ax2.scatter(transMapPreview[:, 1], transMapPreview[:, 0])
    ax2.set_ylim(-testEnv.previewDistance, testEnv.previewDistance)
    ax2.set_xlim(-testEnv.previewDistance, testEnv.previewDistance)

    # pdb.set_trace()

    # prevLen = 10
    prevLen = historyLength*10  # recFreq: 10
    predLen = 20
    ax3.plot(range(1, predLen+1), predictions[:, 0], 'r.-', label='Pred')
    ax3.plot(range(-prevLen, 0), histSpeeds_simul, 'r.-', label='histPred')
    ax3.plot(range(1, predLen+1), trueSpeeds, 'b.-', label='True')
    ax3.plot(range(-prevLen, 0), histSpeeds, 'b.-', label='histTrue')
    ax3.plot(0, curSpeed, 'rx-')
    ax3.plot(0, realCurrentSpeed, 'bx-')
    ax3.set_xlim(-prevLen-1, predLen+2)
    # ax4.set_ylim(0, 210)

    # pdb.set_trace()
    ax5.plot(range(0, predLen), predictions[:, 1], 'r.-', label='Pred')  # at부터 예측하는 거니까
    ax5.plot(range(-prevLen, 0), histAccelX_simul, 'r.-', label='histPred')
    ax5.plot(range(1, predLen+1), trueAccelXs[:-1], 'b.-', label='True')
    # ax5.plot(range(1, predLen+1), trueAccelXs, 'b.-', label='True')  #v31, 32: when historyLength is 0
    ax5.plot(range(-prevLen, 0), histAccelX, 'b.-', label='histTrue')
    ax5.plot(0, curAccelX, 'rx-')  # 사실 이 값도 예측되는 값임.
    ax5.plot(0, realCurrentAccelX, 'bx-')
    ax5.set_xlim(-prevLen-1, predLen+2)

    ax4.plot(curvatures, 'k')  # curvature[0]: 0m (current position)
    ax4.plot(range(len(curvatures)), [0]*len(curvatures), 'k--')
    ax4.set_ylim(-0.05, 0.05)
    ax4a = ax4.twinx()

    # pdb.set_trace()
    # ax4a.plot(np.repeat(np.mean(alphas, axis=0), 2, axis=0), 'r')
    ax4a.plot(np.repeat(alphas[0], 2, axis=0), label='0')
    ax4a.plot(np.repeat(alphas[5], 2, axis=0), label='5')
    ax4a.plot(np.repeat(alphas[10], 2, axis=0), label='10')
    ax4a.plot(np.repeat(alphas[15], 2, axis=0), label='15')
    ax4a.plot(np.repeat(alphas[19], 2, axis=0), label='19')
    ax4a.legend()
    # ax4a.plot(np.repeat(np.mean(alphas, axis=0), 15, axis=0), 'r')
    # ax4a.plot(np.mean(alphas, axis=0), 'r')
    # ax4a.set_ylim(0, 1)
    # ax4a.set_ylim(0, 0.06)  # 175인 경우, 총 35개여서, 1/35보다 조금 큰 값으로 함.
    ax4a.set_ylim(-0.001, 0.02)  # local attention

    # manager.resize(*manager.window.maxsize())  # error
    # manager.frame.Maximize(True)  # error
    # pdb.set_trace()
    vMape = np.mean(abs(predictions[:, 0]-trueSpeeds)/trueSpeeds*100)
    vRmse = np.sqrt(np.mean((predictions[:, 0]-trueSpeeds)**2))
    vCorr = scipy.stats.pearsonr(predictions[:, 0], trueSpeeds)[0]

    plt.savefig('{}/{}_{}_{}_{}.png'.format(resultPath, idx, vMape, vRmse, vCorr))
    plt.close()

    return vMape, vRmse, vCorr

def main():
    # total_curvs = []
    total_preds = []
    total_trues = []
    total_alphas = []

    lmapes = []
    lrmses = []
    # vcorrs = []

    histFlag = False
    # for i in tqdm(range(1500)):
    # for i in tqdm(range(2200)):
    i = 0
    while i<2200:
        print('{:4d}/2200'.format(i), end='\r')
        # if i < 462:
            # pdb.set_trace()
            # continue

        try:
            curPosition, mapPreview, transMapPreview, dist, curvatures, curSpeed, histSpeeds, curAccelX, histAccelX, targetSpeeds, targetaccelXs, histOffsets, curOffset, targetOffsets = testDataStream.__next__()

            if historyLength > 0 and len(histOffsets)==0:
                histFlag = True  # histSpeeds가 없으면, 충분한 hist가 들어올 때까지 histFlag를 True로 했다가, hist가 들어오면, 그 때부터 inference를 하고, flag를 다시 False로.
                continue

            # realCurrentSpeed = curSpeed
            # realCurrentAccelX = curAccelX
            realCurrentOffset = curOffset

            if args.mode == '1': # 만약 model에서 나오는 속력으로만 제어를 한다면, true currentSpeed를 받을 수 없음!
                if i>0 and not histFlag:  # 온전히 모델로만 주행속도 예측
                    # curSpeed = predictions[0][0]
                    # curAccelX = predictions[0][1]
                    curOffset = predictions[0]

                    if historyLength > 0:
                        # histSpeeds_simul.pop(0)
                        # histAccelX_simul.pop(0)
                        # histSpeeds_simul += [currentSpeed_prev]
                        # histAccelX_simul += [currentAccelX_prev]
                        histOffsets_simul.pop(0)
                        histOffsets_simul += [currentOffset_prev]

                if i==0 or histFlag:  # circuit 중에서 대충 중간에서 끊어서 망정이지, 처음부터 시작하면 hist들이 없으므로, _simul을 정하는데 문제가 생길 수 있음!
                    try:
                        # histSpeeds_simul = histSpeeds.tolist()
                        # histAccelX_simul = histAccelX.tolist()  # 실제로 사용은 안함.
                        histOffsets_simul = histOffsets.tolist()
                    except:
                        # histSpeeds_simul = histSpeeds
                        # histAccelX_simul = histAccelX
                        histOffsets_simul = histOffsets
                    histFlag = False

            else: # 학습 자체만 볼때
                # histSpeeds_simul = histSpeeds
                # histAccelX_simul = histAccelX
                histOffsets_simul = histOffsets

        except StopIteration:
            print('\nNo More Data')

            print('The end of the test')
            i = 2200

        # pdb.set_trace()
        # aPreds, alphas = testModel.predict(curvatures[1:1+curvatureLength], curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul)
        Preds, alphas = testModel.predict(curvatures[1:1+curvatureLength], curOffset, histOffsets_simul, curAccelX, histAccelX)  # histAccelX는 사용 안함.
        # curvature[0]: 1m ahead가 되어야 함.
        # 그리고 여기서 curAccelX, histAccelX_simul은 안쓰임.

        # predictions = np.vstack((vPreds, aPreds)).T  # shape (20, 2)

        # if i%5 == 0:
        #     vMape, vRmse, vCorr = draw(i, mapPreview[:curvatureLength], curPosition, transMapPreview[:curvatureLength], curvatures[:curvatureLength], alphas,
        #             curSpeed, realCurrentSpeed, predictions, targetSpeeds, histSpeeds_simul[-histLen:], histSpeeds[-histLen:],
        #             curAccelX, realCurrentAccelX, targetaccelXs, histAccelX_simul[-histLen:], histAccelX[-histLen:])

        total_preds.append(Preds)
        total_trues.append(targetOffsets)
        total_alphas.append(np.repeat(np.mean(alphas, axis=0), 2, axis=0))

        lMape = np.mean(abs(targetOffsets-Preds)/abs(targetOffsets)*100)
        lRmse = np.sqrt(np.sum((targetOffsets-Preds)**2))

        lmapes.append(lMape)
        lrmses.append(lRmse)
        # vcorrs.append(vCorr)

        # currentSpeed_prev = curSpeed
        # currentAccelX_prev = curAccelX
        currentOffset_prev = curOffset

        i += 1

    print('Avg lMAPE: {}'.format(np.mean(np.array(lmapes))))
    print('Avg lRMSE: {}'.format(np.mean(np.array(lrmses))))
    # print('Avg vCORR: {}'.format(np.mean(np.array(vcorrs))))

    np.save('{}/total_preds'.format(resultPath), np.array(total_preds))
    np.save('{}/total_trues'.format(resultPath), np.array(total_trues))
    np.save('{}/total_alphas'.format(resultPath), np.array(total_alphas))


if __name__ == "__main__":
    main()
