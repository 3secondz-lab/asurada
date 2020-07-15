import pdb
from test_data import *
from test_model import *

import numpy as np
import matplotlib.pyplot as plt

import time
import argparse
import scipy

# driverNum = 1  # (1, 7), (4, 7), (5, 5)

# testNum = 4
# testEnv = TestEnv('../Data/IJF_D{}/std_00{}.csv'.format(driverNum, testNum), recFreq=20, cWindow=10.0, vpWindow=2)
# print('Test1; Driver {}'.format(driverNum), 'and TestFileNum {}'.format(testNum))

#testEnv = test_data2.TestEnv('../../debug')
#print('Test2; Driver {}'.format(driverNum))

# testEnv = TestEnv('../Data/003_0422_ij/2_stdFile/set4-2.csv',
#                  recFreq=10, cWindow=10.0, vpWindow=2, condition=True)
# testEnv = TestEnv('../Data/003_0420_yy/2_stdFile/set1.csv',
#                   recFreq=10, cWindow=10.0, vpWindow=2, condition=True)

''' ex. python test_main.py -c imola '''

parser = argparse.ArgumentParser()
parser.add_argument('--circuit', '-c')

args = parser.parse_args()
assert args.circuit in ['mugello', 'imola', 'magione', 'spa'], 'Valid circuit: mugello, imola, magione, spa'

if args.circuit == 'mugello':  # 전체 데이터를 다보려면, 너무 시간이 오래 걸려서 임의로 2바퀴 정도를 지정함.
    st=1593005782
    et=1593006102
elif args.circuit == 'imola':
    st=1593006843
    et=1593007142
elif args.circuit == 'magione':
    st=1593000712
    et=1593000900
elif args.circuit == 'spa':
    st=1593002550
    et=1593002939

datafile = '../Data/YJ/{}-recoder-scz_msgs.csv'.format(args.circuit)
mapfile = '../Data/YJ/{}.csv'.format(args.circuit)
testEnv = TestEnv(datafile,
                 recFreq=10, cWindow=100.0, vpWindow=2, condition=True,
                 st=st, et=et, mapfile=mapfile)

testDataStream = testEnv.get_preview()  # preview를 입력받는 상황을 가정

#dataName = 'IJF_D{}_20_2_2_10_10_10'.format(driverNum)  # cWindow [s], vWindow [s], vpWindow [s], cUnit [Hz], vUnit [Hz], vpUnit [Hz]
#dataName_encc = 'IJF_D{}_20_2_2_10_10_10'.format(driverNum_encc)
#dataName = 'IJ_20_2_2_10_10_10'
dataName = 'YJ_TD_100_2_2_10_10_10'

dataName_encc = dataName

epoch = 190

# chpt_encC_path = './BEST_checkpoint_ENCC_{}_{}.pth.tar'.format(dataName_encc, epoch)
# chpt_encD_path = './BEST_checkpoint_ENCD_{}_{}.pth.tar'.format(dataName, epoch)
# chpt_dec_path = './BEST_checkpoint_DEC_{}_{}.pth.tar'.format(dataName, epoch)
# chpt_stat_path ='./BEST_stat_{}_{}.pickle'.format(dataName, epoch)

# chpt_encC_path = './checkpoint_ENCC_{}.pth.tar'.format(dataName)
# chpt_encD_path = './checkpoint_ENCD_{}.pth.tar'.format(dataName)
# chpt_dec_path = './checkpoint_DEC_{}.pth.tar'.format(dataName)
# chpt_stat_path = './stat_{}.pickle'.format(dataName)

chpt_encC_path = './chpt_yj/checkpoint_ENCC_{}_{}.pth.tar'.format(dataName_encc, epoch)
chpt_encD_path = './chpt_yj/checkpoint_ENCD_{}_{}.pth.tar'.format(dataName, epoch)
chpt_dec_path = './chpt_yj/checkpoint_DEC_{}_{}.pth.tar'.format(dataName, epoch)
chpt_stat_path ='./chpt_yj/stat_{}_{}.pickle'.format(dataName, epoch)

testModel = Model(chpt_encC_path, chpt_encD_path, chpt_dec_path, chpt_stat_path)  # speed prediction에 사용할 model parameters


def main():
    # speedDiffLimit = 6.5  # km/h
    # speedDiffSet = 6.5  # km/h

    realCurrentSpeeds = []
    currentSpeeds = []
    predSpeeds = []
    # diffs_p = []
    # diffs_m = []

    total_curvs = []
    total_preds = []
    total_alphas_d = []
    total_alphas_c = []

    times = []

    # currentSpeed = 100  # 임시 시작 스피드
    i = 0
    while(True):
        try:
            ## 현재 속도에 따른 preview distance를 다르게 해보려고 임시로 추가
            ## adaptive preview distance according to current speed
            # previewDistance = currentSpeeds / 3600 * 10
            # print('\t{}'.format(previewDistance))
            # testEnv.dh.set_preview_distance(currentSpeed * 1000 / 3600 * 10)

            curvatures, currentSpeed, histSpeeds = testDataStream.__next__()  # read at recFreq
            # curvatures, histSpeeds :np.ndarray
            # currentSpeed: np.float

            realCurrentSpeed = currentSpeed

            # 만약 model에서 나오는 속력으로만 제어를 한다면, true currentSpeed를 받을 수 없음!
            if i>0:  # 온전히 모델로만 주행속도 예측
                currentSpeed = preds.squeeze()[0].item()

        except StopIteration:
            print('No More Data')
            return realCurrentSpeeds, currentSpeeds, predSpeeds, times, total_preds, total_alphas_c, total_alphas_d, total_curvs

        st = time.time()
        preds, alphas_d, alphas_c = testModel.predict(curvatures, currentSpeed, histSpeeds)
        times.append(time.time() - st)

        pred = preds.squeeze()[9].item()  # predicted speed 1s ahead, 0.1초 간격으로 예측하도록 함, [0]: 0.1초 후 예측 속력

        # diff = currentSpeed - pred  # 추후 예측 속도 smoothing에 사용할 정보
        # if diff > 0:
        #     diffs_m.append(diff)
        # else:
        #     diffs_p.append(diff)

        # if abs(currentSpeed - pred) > speedDiffLimit:  # 예측 속도 임시 smoothing
        #     if currentSpeed < pred:  # 가속구간
        #         pred = currentSpeed + speedDiffSet
        # #     else:  # 감속구간
        # #         pred = currentSpeed - speedDiffSet

        realCurrentSpeeds.append(realCurrentSpeed)  # 실제 주행 데이터
        currentSpeeds.append(currentSpeed)  # [0]만 실제 주행 데이터, [1:]부터는 계속 예측된 데이터
        predSpeeds.append(pred)

        total_preds.append(preds)
        total_alphas_c.append(alphas_c)
        total_alphas_d.append(alphas_d)
        total_curvs.append(curvatures)

        i+=1
        if i>0 and i%100 == 0:
            print(i)

if __name__ == "__main__":
    realTrue, predCurrent, pred, times, tt_pred, tt_ac, tt_ad, tt_cv = main()

    recFreq = 10  # 데이터의 unit에 맞게 여기도 바꾸어야 함!

    d_true = np.array(realTrue[recFreq:])
    d_pred = np.array(pred[:-recFreq])  # d_true랑 길이 맞추려고

    predCurrent = np.array(predCurrent)  # 이건 그냥 온전히 저장

    rmse = np.sqrt(sum((d_true - d_pred)**2)/len(d_true))
    mape = 100*sum(abs(d_true - d_pred)/d_true)/len(d_true)

    print('Average elapsed time: {:4f}'.format(sum(times)/len(times)))
    print('MAPE: {}'.format(mape))
    print('RMSE: {}'.format(rmse))

    pearson, pv = scipy.stats.pearsonr(d_true, d_pred)
    print('Pearson Correlation: {} (pValue: {})'.format(pearson, pv))

    plt.plot(d_true, 'b', label='True')
    plt.plot(d_pred, 'r', label='Pred')
    plt.legend()
    plt.title('Predict speed 1s ahead (MAPE: {:.3f}, RMSE: {:.3f}, Corr: {:.3f} (pv: {:.3f}))'.format(mape, rmse, pearson, pv))
    plt.show()

    # pdb.set_trace()

    # np.save('true_d{}'.format(driverNum), realTrue)  # realTrue[recFreq]과
    # np.save('pred_d{}'.format(driverNum), pred)  # pred[0] 가 서로 짝임.
    # np.save('curpred_d{}'.format(driverNum), predCurrent)  # realTrue[0]과 predCurrent[0]이 서로 짝임.
    #
    # np.save('tt_pred_d{}'.format(driverNum),
    #         np.array([x.squeeze().detach().cpu().numpy() for x in tt_pred]))
    # np.save('tt_ac_d{}'.format(driverNum),
    #         np.array([torch.mean(x.squeeze(), dim=0).detach().cpu().numpy() for x in tt_ac][:-1]))
    # np.save('tt_ad_d{}'.format(driverNum),
    #         np.array([torch.mean(x.squeeze(), dim=0).detach().cpu().numpy() for x in tt_ad]))
    # np.save('tt_cv_d{}'.format(driverNum),
    #         np.array(tt_cv[:-1]))
