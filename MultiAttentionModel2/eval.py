import pdb

import argparse
import pickle

from model import Encoder, Decoder
from dataset import Dataset
from constants import device

import torch
from tqdm import tqdm
from random import random
import matplotlib.pyplot as plt
import numpy as np

from utils import *

# Model parameters
input_size = 1
hidden_size = 64
output_size = 1

# Testing parameters
batch_size = 128
workers = 0

def evaluate(driverNum, epoch, loaderType, vis=False):
    # dataName = 'IJF_D{}_20_2_2_10_10_10'.format(driverNum)
    # dataName = 'IJ_20_2_2_10_10_10'
    dataName = 'YJ_TD_100_2_2_10_10_10'  # # cWindow [s], vWindow [s], vpWindow [s], cUnit [Hz], vUnit [Hz], vpUnit [Hz]

    # chpt_encC_path = './BEST_checkpoint_ENCC_{}.pth.tar'.format(dataName)
    # chpt_encD_path = './BEST_checkpoint_ENCD_{}.pth.tar'.format(dataName)
    # chpt_dec_path = './BEST_checkpoint_DEC_{}.pth.tar'.format(dataName)
    # chpt_stat_path ='./BEST_stat_{}.pickle'.format(dataName)

    # chpt_encC_path = './checkpoint_ENCC_{}.pth.tar'.format(dataName)
    # chpt_encD_path = './checkpoint_ENCD_{}.pth.tar'.format(dataName)
    # chpt_dec_path = './checkpoint_DEC_{}.pth.tar'.format(dataName)
    # chpt_stat_path = './stat_{}.pickle'.format(dataName)

    chpt_encC_path = './chpt_yj/checkpoint_ENCC_{}_{}.pth.tar'.format(dataName, epoch)
    chpt_encD_path = './chpt_yj/checkpoint_ENCD_{}_{}.pth.tar'.format(dataName, epoch)
    chpt_dec_path = './chpt_yj/checkpoint_DEC_{}_{}.pth.tar'.format(dataName, epoch)
    chpt_stat_path ='./chpt_yj/stat_{}_{}.pickle'.format(dataName, epoch)

    with open(chpt_stat_path, 'rb') as f:
        chpt_stat = pickle.load(f)

    encoder_c = Encoder(input_size=input_size, enc_dim=hidden_size)
    encoder_d = Encoder(input_size=input_size, enc_dim=hidden_size)
    decoder = Decoder(enc_dim=hidden_size, dec_dim=hidden_size, att_dim=hidden_size, output_dim=output_size)

    encoder_c.load_state_dict(torch.load(chpt_encC_path))
    encoder_d.load_state_dict(torch.load(chpt_encD_path))
    decoder.load_state_dict(torch.load(chpt_dec_path))

    encoder_c = encoder_c.to(device)
    encoder_d = encoder_d.to(device)
    decoder = decoder.to(device)

    encoder_c.eval()
    encoder_d.eval()
    decoder.eval()

    testLoader = torch.utils.data.DataLoader(Dataset(dataName, loaderType,
        cMean=chpt_stat['cMean_tr'], cStd=chpt_stat['cStd_tr'],
        sMean=chpt_stat['sMean_tr'], sStd=chpt_stat['sStd_tr']),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    mapes = []
    rmses = []

    mapes_1s = []
    rmses_1s = []

    cMean = chpt_stat['cMean_tr']
    cStd = chpt_stat['cStd_tr']

    mean = chpt_stat['sMean_tr']
    std = chpt_stat['sStd_tr']

    # for i, (curvatures, targetSpeeds, histSpeeds) in enumerate(tqdm(testLoader)):
    for i, (curvatures, targetSpeeds, histSpeeds) in enumerate(testLoader):
        curvatures = curvatures.to(device)
        targetSpeeds = targetSpeeds.to(device)
        histSpeeds = histSpeeds.to(device)

        decodeLength = 20  # 2sec.

        enc_hiddens_c = encoder_c(curvatures)
        enc_hiddens_v = encoder_d(histSpeeds)
        predictions, alphas_d, alphas_c = decoder(enc_hiddens_v, enc_hiddens_c, targetSpeeds[:, 0], decodeLength)

        targets = targetSpeeds[:, 1:]

        mape, rmse = accuracy(predictions*std+mean, targets*std+mean)
        mapes.append(mape)
        rmses.append(rmse)

        mape_1s, rmse_1s = accuracy(predictions[:, 9]*std+mean, targets[:, 9]*std+mean)
        mapes_1s.append(mape_1s)
        rmses_1s.append(rmse_1s)

        if vis:
            if random() > 0.6:
                fig, (ax1, ax2) = plt.subplots(2, 1)

                idx = 0

                curvature = (curvatures[idx]*cStd+cMean).detach().cpu().numpy()  # 201
                pred = (predictions[idx]*std+mean).detach().cpu().numpy()  # 20
                true = (targetSpeeds[idx]*std+mean).detach().cpu().numpy()  # 21
                alpha = torch.mean(alphas_c[idx], dim=0).detach().cpu().numpy()
                # (20, 201)  # 0.1초 간격으로 2초 동안 예측했을 때,
                # 매 예측 포인트에서 참조된 alpha 값을 가지고 있으나,
                # 2초 내에서 alpha의 값은 크게 변화하지 않는 것 같으므로,
                # 그냥 mean을 때려버림.

                currentCurvature = curvature[0]
                previewCurvature = curvature[1:]
                currentSpeed = true[0]
                targetSpeeds = true[1:]

                mape, rmse = accuracy(predictions[idx]*std+mean, targets[idx]*std+mean)

                ax1.plot(np.arange(1, len(previewCurvature)+1), abs(previewCurvature), 'k--', label='curvature')
                ax1a = ax1.twinx()
                ax1a.plot(np.arange(1, len(previewCurvature)+1), alpha[1:], label='alpha_c')
                ax1.legend()
                ax1a.legend()

                ax2.scatter(0, currentSpeed, color='b', marker='x')
                ax2.plot(np.insert(pred, 0, currentSpeed, axis=0), 'r.-', label='pred')
                ax2.plot(np.insert(targetSpeeds, 0, currentSpeed, axis=0), 'b.-', label='true')
                ax2.legend()

                ax1.set_title('MAPE:{:.4f}, RMSE:{:.4f}'.format(mape, rmse))
                plt.show()

    avgMAPE = sum(mapes)/len(mapes)
    avgRMSE = sum(rmses)/len(rmses)

    avgMAPE_1s = sum(mapes_1s)/len(mapes_1s)
    avgRMSE_1s = sum(rmses_1s)/len(rmses_1s)

    return avgMAPE, avgRMSE, avgMAPE_1s, avgRMSE_1s

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dn', '-dn')
    parser.add_argument('--vis', '-v')

    args = parser.parse_args()

    # epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    epochs = [*range(10, 200, 10)]
    mapes = {'TRAIN':[], 'VAL':[], 'TEST':[]}
    rmses = {'TRAIN':[], 'VAL':[], 'TEST':[]}
    for epoch in epochs:
        print('\n{}'.format(epoch))
        for loaderType in ['TRAIN', 'VAL', 'TEST']:
            print('\n', loaderType)

            mape, rmse, mape_1s, rmse_1s = evaluate(args.dn, epoch, loaderType, args.vis)

            print('\nTest Performance')
            print('MAPE: {:.4}'.format(mape))
            print('RMSE: {:.4}'.format(rmse))
            print('\nMAPE(1s): {:.4}'.format(mape_1s))
            print('RMSE(1s): {:.4}'.format(rmse_1s))

            mapes[loaderType].append(mape)
            rmses[loaderType].append(rmse)

    plt.figure()
    plt.plot(epochs, mapes['TRAIN'], 'b.-', label='TRAIN')
    plt.plot(epochs, mapes['VAL'], 'g.-', label='VAL')
    plt.plot(epochs, mapes['TEST'], 'r.-', label='TEST')
    plt.legend()
    plt.grid()
    plt.show()
