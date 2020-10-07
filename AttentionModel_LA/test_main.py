import pdb
from test_data import *
from test_model import *

import numpy as np
import matplotlib.pyplot as plt

import time
import scipy
import os
from tqdm import tqdm

from test_constants import *

# rescale = False

if args.circuit == 'mugello':  # 전체 데이터를 다보려면 시간이 너무 오래걸려서, 임의 구간 지정
    st=1593005782
    et=1593006102
elif args.circuit == 'imola':
    st=1593006843
    et=1593007142
elif args.circuit == 'magione':
    et=1593000900
    st=1593000712
elif args.circuit == 'spa':
    st=1593002550
    et=1593002939
elif args.circuit == 'nord':
    args.circuit = 'nordschleife'
    st=1595326774
    et=1595327336
elif args.circuit == 'YYF':
    st=200  #50
    et=550  #200
    # rescale = True
    # rescale_rate = 0.2

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

    prevLen = historyLength*10  # recFreq: 10
    predLen = predLength
    ax3.plot(range(1, predLen+1), predictions[:, 0], 'r.-', label='Pred')
    ax3.plot(range(-prevLen, 0), histSpeeds_simul, 'r.-', label='histPred')
    ax3.plot(range(1, predLen+1), trueSpeeds, 'b.-', label='True')
    ax3.plot(range(-prevLen, 0), histSpeeds, 'b.-', label='histTrue')
    ax3.plot(0, curSpeed, 'rx-')
    ax3.plot(0, realCurrentSpeed, 'bx-')
    ax3.set_xlim(-prevLen-1, predLen+2)

    ax5.plot(range(0, predLen), predictions[:, 1], 'r.-', label='Pred')  # at부터 예측하는 거니까
    ax5.plot(range(-prevLen, 0), histAccelX_simul, 'r.-', label='histPred')
    ax5.plot(range(1, predLen+1), trueAccelXs[:-1], 'b.-', label='True')
    # ax5.plot(range(1, predLen+1), trueAccelXs, 'b.-', label='True')  #v31, 32: when historyLength is 0
    ax5.plot(range(-prevLen, 0), histAccelX, 'b.-', label='histTrue')
    ax5.plot(0, curAccelX, 'rx-')
    ax5.plot(0, realCurrentAccelX, 'bx-')
    ax5.set_xlim(-prevLen-1, predLen+2)

    ax4.plot(curvatures, 'k')  # curvature[0]: 0m (current position)
    ax4.plot(range(len(curvatures)), [0]*len(curvatures), 'k--')
    ax4.set_ylim(-0.05, 0.05)
    ax4a = ax4.twinx()

    if predLength == 5:
        ax4a.plot(np.repeat(alphas[0], 2, axis=0), label='0')
        ax4a.plot(np.repeat(alphas[4], 2, axis=0), label='4')
    else: #20
        ax4a.plot(np.repeat(alphas[0], 2, axis=0), label='0')
        ax4a.plot(np.repeat(alphas[5], 2, axis=0), label='5')
        ax4a.plot(np.repeat(alphas[10], 2, axis=0), label='10')
        ax4a.plot(np.repeat(alphas[15], 2, axis=0), label='15')
        ax4a.plot(np.repeat(alphas[19], 2, axis=0), label='19')
    ax4a.legend()
    ax4a.set_ylim(-0.001, alpha_ylimMax)  # local attention

    vMape = np.mean(abs(predictions[:, 0]-trueSpeeds)/trueSpeeds*100)
    vRmse = np.sqrt(np.mean((predictions[:, 0]-trueSpeeds)**2))
    vCorr = scipy.stats.pearsonr(predictions[:, 0], trueSpeeds)[0]

    plt.savefig('{}/{}_{}_{}_{}.png'.format(resultPath, idx, vMape, vRmse, vCorr))
    plt.close()

    return vMape, vRmse, vCorr

def main():
    total_preds = []
    total_trues = []
    total_alphas = []

    vmapes = []
    vrmses = []
    vcorrs = []

    histFlag = False
    for i in tqdm(range(1500)):
        try:
            curPosition, mapPreview, transMapPreview, dist, curvatures, curSpeed, histSpeeds, curAccelX, histAccelX, targetSpeeds, targetaccelXs = testDataStream.__next__()

            if historyLength > 0 and len(histSpeeds)==0:
                histFlag = True  # histSpeeds가 없으면, 충분한 hist가 들어올 때까지 histFlag를 True로 했다가, hist가 들어오면, 그 때부터 inference를 하고, flag를 다시 False로.
                continue

            realCurrentSpeed = curSpeed
            realCurrentAccelX = curAccelX

            if args.mode == '1': # 만약 model에서 나오는 속력으로만 제어를 한다면, true currentSpeed를 받을 수 없음!
                if i>0 and not histFlag:  # 온전히 모델로만 주행속도 예측
                    curSpeed = predictions[0][0]
                    curAccelX = predictions[0][1]

                    if historyLength > 0:
                        histSpeeds_simul.pop(0)
                        histAccelX_simul.pop(0)
                        histSpeeds_simul += [currentSpeed_prev]
                        histAccelX_simul += [currentAccelX_prev]

                if i==0 or histFlag:  # hist speed가 없는 처음부터 시작하면 hist들이 없으므로, _simul을 정하는데 문제가 생길 수 있음!
                    try:
                        histSpeeds_simul = histSpeeds.tolist()
                        histAccelX_simul = histAccelX.tolist()
                    except:
                        histSpeeds_simul = histSpeeds
                        histAccelX_simul = histAccelX
                    histFlag = False

            else: # 학습 자체 환경과 똑같은 환경에서 test, m=0
                histSpeeds_simul = histSpeeds
                histAccelX_simul = histAccelX

        except StopIteration:
            print('No More Data')

            print('Avg vMAPE: {}'.format(np.mean(np.array(vmapes))))
            print('Avg vRMSE: {}'.format(np.mean(np.array(vrmses))))
            print('Avg vCORR: {}'.format(np.mean(np.array(vcorrs))))

            np.save('{}/total_preds'.format(resultPath), np.array(total_preds))
            np.save('{}/total_trues'.format(resultPath), np.array(total_trues))
            np.save('{}/total_alphas'.format(resultPath), np.array(total_alphas))

            for j in range(len(total_preds)):
                plt.plot(range(j, j+predLength), total_preds[j][:, 0], 'k.-', alpha=0.1)
                plt.plot(range(j, j+predLength), total_trues[j], 'r.-')
            plt.show()

        aPreds, alphas = testModel.predict(curvatures[1:1+curvatureLength], curSpeed, histSpeeds_simul, curAccelX, histAccelX_simul)

        # if rescale:
            # vPreds = np.cumsum(aPreds * rescale_rate) + curSpeed
        # else:
        vPreds = np.cumsum(aPreds) + curSpeed

        predictions = np.vstack((vPreds, aPreds)).T  # shape (20, 2)

        if i%5 == 0:
            vMape, vRmse, vCorr = draw(i, mapPreview[:curvatureLength], curPosition, transMapPreview[:curvatureLength], curvatures[:curvatureLength], alphas,
                    curSpeed, realCurrentSpeed, predictions, targetSpeeds, histSpeeds_simul[-histLen:], histSpeeds[-histLen:],
                    curAccelX, realCurrentAccelX, targetaccelXs, histAccelX_simul[-histLen:], histAccelX[-histLen:])

        total_preds.append(predictions)
        total_trues.append(targetSpeeds)
        total_alphas.append(np.repeat(np.mean(alphas, axis=0), 2, axis=0))

        vmapes.append(vMape)
        vrmses.append(vRmse)
        vcorrs.append(vCorr)

        currentSpeed_prev = curSpeed
        currentAccelX_prev = curAccelX

    print('Avg vMAPE: {}'.format(np.mean(np.array(vmapes))))
    print('Avg vRMSE: {}'.format(np.mean(np.array(vrmses))))
    print('Avg vCORR: {}'.format(np.mean(np.array(vcorrs))))

    np.save('{}/total_preds'.format(resultPath), np.array(total_preds))
    np.save('{}/total_trues'.format(resultPath), np.array(total_trues))
    np.save('{}/total_alphas'.format(resultPath), np.array(total_alphas))

    for j in range(len(total_preds)):
        plt.plot(range(j, j+predLength), total_preds[j][:, 0], 'k.-', alpha=0.1)
        plt.plot(range(j, j+predLength), total_trues[j], 'r.-')
    plt.show()

    pdb.set_trace()  # press c to continue


if __name__ == "__main__":
    main()
