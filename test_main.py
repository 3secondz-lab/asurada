import pdb
from test_data import *
from test_model import *

import numpy as np
import matplotlib.pyplot as plt

testEnv = TestEnv('./Data/std_003.csv', recFreq=20, previewT=10.0)
testDataStream = testEnv.get_preview()  # preview를 입력받는 상황을 가정
testModel = Model('./learned/21300.pth')  # speed prediction에 사용할 model parameters

def main():
    speedDiffLimit = 6.5  # km/h
    speedDiffSet = 6.5  # km/h

    currentSpeeds = []
    predSpeeds = []
    # diffs_p = []
    # diffs_m = []

    i = 0
    while(True):
        try:
            curvatures, currentSpeed, currentThrottle, currentSteer = testDataStream.__next__()
            # 현재 currentThrottle과 currentSteer는 사용되지 않음
            # curvatures는 numpy array vector, currentSpeed는 scalar
        except StopIteration:
            print('No More Data')
            return currentSpeeds, predSpeeds

        preds = testModel.predict(curvatures, currentSpeed, currentThrottle, currentSteer)
        pred = preds[20].item()  # predicted speed 1s ahead (record unit: 0.05)

        # diff = currentSpeed - pred  # 추후 예측 속도 smoothing에 사용할 정보
        # if diff > 0:
        #     diffs_m.append(diff)
        # else:
        #     diffs_p.append(diff)

        if abs(currentSpeed - pred) > speedDiffLimit:  # 예측 속도 임시 smoothing
            if currentSpeed < pred:  # 가속구간
                pred = currentSpeed + speedDiffSet
        #     else:  # 감속구간
        #         pred = currentSpeed - speedDiffSet

        currentSpeeds.append(currentSpeed)
        predSpeeds.append(pred)

        i+=1
        if i>0 and i%100 == 0:
            print(i)

if __name__ == "__main__":
    true, pred = main()

    true = np.array(true[1:])
    pred = np.array(pred[:-1])

    rmse = np.sqrt(sum((true - pred)**2)/len(true))
    mape = 100*sum(abs(true - pred)/true)/len(true)

    plt.plot(true, 'b', label='True')
    plt.plot(pred, 'r', label='Pred')
    plt.legend()
    plt.title('Predict speed 1s ahead (MAPE: {:.3f}, RMSE: {:.3f})'.format(mape, rmse))
    plt.show()
