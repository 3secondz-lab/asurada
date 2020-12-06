import pdb

import torch
import torch.optim
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, DecoderWithAttention

from dataset import Dataset
from constants import device, hiddenDimension

from utils import *
import matplotlib.pyplot as plt
import scipy
import shutil
from random import random


''' ======================================================================== '''
''' Data '''
driver = 'YJ'  # mugello, spa, imola, magione, YYF
# driver = 'YS'  # mugello, spa, imola, magione, monza

vNumber = 1  # YJ, TEST:SPA
# vNumber = 2  # YJ, TEST:IMOLA
# vNumber = 3  # YJ, TEST:MAGIONE
# vNumber = 4  # YJ, TEST:MUGELLO

# vNumber = 5  # YS, TEST:SPA
# vNumber = 6  # YS, TEST:IMOLA
# vNumber = 7  # YS, TEST:MAGIONE
# vNumber = 8  # YS, TEST:MUGELLO

circuit_tr = ['mugello', 'magione', 'imola']  # vNumber: 1. 5
circuit_vl = ['spa']

# circuit_tr = ['mugello', 'magione', 'spa']  # vNumber: 2, 6
# circuit_vl = ['imola']

# circuit_tr = ['mugello', 'imola', 'spa']  #  vNumber: 3, 7
# circuit_vl = ['magione']

# circuit_tr = ['magione', 'imola', 'spa']  # vNumber: 4, 8
# circuit_vl = ['mugello']

''' Env. Parameters '''
curvatureLength = 250  # unit [m]
historyLength = 10  # unit [0.1s]
predLength = 20  # unit [0.1s]

''' Model '''
encoder_dim = hiddenDimension  # hiddenDimension defined @ constants.py
lstm_input_dim = historyLength + 1  # [history, vt]
decoder_dim = hiddenDimension  # hidden state
attention_dim = hiddenDimension
output_dim = 1  # diff(v)
criterion = MSEwAtt

''' Training parameters '''
start_epoch = 0
epochs = 200  # about 2Hr.
batch_size = 1024
workers = 0  # 0 for windows
encoder_lr = 1e-3
decoder_lr = 1e-3

epochs_since_improvement = 0
best_loss_tr = 10e5
grad_clip = 5.

training_gt_rate = 0  # 0: 예측값으로만 학습
''' ======================================================================== '''

''' System '''
chptFolderPath = './chpt_{}_v{}'.format(driver, vNumber)
if not os.path.exists(chptFolderPath):
    os.mkdir(chptFolderPath)

tensorBoardPath = '{}_Training_{}'.format(driver, vNumber)
if os.path.exists(tensorBoardPath):
    shutil.rmtree(tensorBoardPath)  # Empty directory
    print('Remove {}\n'.format(tensorBoardPath))
writer = SummaryWriter(log_dir=tensorBoardPath)  # ex. tensorboard --logdir YJ_Training_1


def main():
    global epochs_since_improvement, best_loss_tr

    encoder = Encoder()
    decoder = DecoderWithAttention(encoder_dim, lstm_input_dim, decoder_dim, attention_dim, output_dim)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    trainLoader = torch.utils.data.DataLoader(Dataset(driver, circuit_tr,
                                            curvatureLength, historyLength, predLength),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    cMean_tr = trainLoader.dataset.cMean
    cStd_tr = trainLoader.dataset.cStd
    vMean_tr = trainLoader.dataset.vMean
    vStd_tr = trainLoader.dataset.vStd
    aMean_tr = trainLoader.dataset.aMean
    aStd_tr = trainLoader.dataset.aStd

    validLoader = torch.utils.data.DataLoader(Dataset(driver, circuit_vl,
                                            curvatureLength, historyLength, predLength,
                                            cMean=cMean_tr, cStd=cStd_tr, vMean=vMean_tr, vStd=vStd_tr, aMean=aMean_tr, aStd=aStd_tr),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('Training version.{} (A->V)'.format(vNumber))
    print('Training data ({} - {})'.format(driver, circuit_tr))
    print('Validation data ({} - {})'.format(driver, circuit_vl))
    print('curvature len {}'.format(curvatureLength))
    print('history len {}'.format(historyLength))
    print('pred len {}'.format(predLength))
    print('hiddenDimension {}'.format(hiddenDimension))

    print('\nTraining...\n')

    for epoch in tqdm(range(start_epoch, epochs)):

        loss, vMape, vRmse, vCorr, aCorr = train(trainLoader=trainLoader,
                                                 encoder=encoder,
                                                 decoder=decoder,
                                                 criterion=criterion,
                                                 encoder_optimizer=encoder_optimizer,
                                                 decoder_optimizer=decoder_optimizer,
                                                 epoch=epoch)

        writer.add_scalars('Loss', {'tr':loss}, epoch)
        writer.add_scalars('MAPE', {'tr':vMape}, epoch)
        writer.add_scalars('RMSE', {'tr':vRmse}, epoch)
        writer.add_scalars('vCorr', {'tr':vCorr}, epoch)
        writer.add_scalars('aCorr', {'tr':aCorr}, epoch)

        is_best = loss < best_loss_tr
        best_loss_tr = min(loss, best_loss_tr)
        if not is_best:
            epochs_since_improvement += 1
            print('\nEpoch {} Epoch Epochs since last improvement (unit: 100): {}\n'.format(epoch, epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(epoch, encoder_optimizer, 0.8)
            adjust_learning_rate(epoch, decoder_optimizer, 0.8)

        if epoch%5 == 0:
            loss_vl, vMape_vl, vRmse_vl, vCorr_vl, aCorr_vl = validate(validLoader=validLoader,
                                                                        encoder=encoder,
                                                                        decoder=decoder,
                                                                        criterion=criterion)
            writer.add_scalars('Loss', {'vl':loss_vl}, epoch)
            writer.add_scalars('MAPE', {'vl':vMape_vl}, epoch)
            writer.add_scalars('RMSE', {'vl':vRmse_vl}, epoch)
            writer.add_scalars('vCorr', {'vl':vCorr_vl}, epoch)
            writer.add_scalars('aCorr', {'vl':aCorr_vl}, epoch)

        if epoch%10 == 0:
            save_checkpoint(chptFolderPath, encoder, decoder, epoch, cMean_tr, cStd_tr, vMean_tr, vStd_tr, aMean_tr, aStd_tr, curvatureLength, historyLength)
    writer.close()

stepIdx = 0
def train(trainLoader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):

    global stepIdx

    encoder.train()
    decoder.train()

    vMean = trainLoader.dataset.vMean
    vStd = trainLoader.dataset.vStd

    aMean = trainLoader.dataset.aMean
    aStd = trainLoader.dataset.aStd

    mean = torch.Tensor([vMean, aMean]).to(device)
    std = torch.Tensor([vStd, aStd]).to(device)

    losses = []
    vMapes = []
    vRmses = []
    vCorrs = []
    aCorrs = []

    for i, (curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs) in enumerate(trainLoader):

        curvatures = curvatures.to(device)  # [0]: 1m ahead
        targetSpeeds = targetSpeeds.to(device)  # [0]: currentSpeed, v_t
        histSpeeds = histSpeeds.to(device)
        targetAccelXs = targetAccelXs.to(device)  # [0]: a_t
        histAccelXs = histAccelXs.to(device)

        decodeLength = targetSpeeds.size(1)-1  # [0]: currentSpeed

        curvatures = encoder(curvatures.unsqueeze(dim=1))

        if random() >= training_gt_rate:
            predictions, alphas, alphas_target = decoder(curvatures, targetSpeeds[:, 0], histSpeeds, targetAccelXs, histAccelXs, decodeLength,
                                        trainLoader.dataset.vMean, trainLoader.dataset.vStd,
                                        trainLoader.dataset.aMean, trainLoader.dataset.aStd)
        else:
            predictions, alphas, alphas_target = decoder(curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs, decodeLength,
                                        trainLoader.dataset.vMean, trainLoader.dataset.vStd,
                                        trainLoader.dataset.aMean, trainLoader.dataset.aStd)

        targets = targetAccelXs[:, :-1].unsqueeze(-1)

        loss, metrics = criterion(predictions, targets, mean, std, alphas, alphas_target)  # 여기도 정리
        # metrics: MSE, vMSE, aMSE, consistencyVA

        stepIdx += 1
        writer.add_scalars('MSE', {'tr':metrics[0].item()}, stepIdx)
        # writer.add_scalars('vMSE', {'tr':metrics[1].item()}, stepIdx)
        # writer.add_scalars('aMSE', {'tr':metrics[2].item()}, stepIdx)
        # # writer.add_scalars('VS', {'tr':vsmoothing.item()}, stepIdx)
        # # writer.add_scalars('AS', {'tr':asmoothing.item()}, stepIdx)
        # writer.add_scalars('Consistency', {'tr':metrics[3].item()}, stepIdx)
        writer.add_scalars('ATT', {'tr':metrics[4].item()}, stepIdx)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            clip_gradient(encoder_optimizer, grad_clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        aPreds = (predictions * aStd + aMean).squeeze(-1)
        curSpeed = targetSpeeds[:, 0]*vStd+vMean
        vPreds = torch.cumsum(aPreds, dim=1) + (curSpeed).unsqueeze(-1)
        vTrues = targetSpeeds[:, 1:]*vStd+vMean
        aTrues = targetAccelXs[:, :-1]*aStd+aMean

        vMape, vRmse = accuracy(vPreds, vTrues)
        vCorr, aCorr = pearsonr(torch.cat((vPreds.unsqueeze(-1), aPreds.unsqueeze(-1)), 2), torch.cat((vTrues.unsqueeze(-1), aTrues.unsqueeze(-1)), 2))

        losses.append(loss.item())
        vMapes.append(vMape)
        vRmses.append(vRmse)
        vCorrs.append(vCorr)
        aCorrs.append(aCorr)

    return np.mean(np.array(losses)), np.mean(np.array(vMapes)), np.mean(np.array(vRmses)), np.mean(np.array(vCorrs)), np.mean(np.array(aCorrs))

def validate(validLoader, encoder, decoder, criterion):  # training_gt_rate --> 0
    encoder.eval()
    decoder.eval()

    vMean = validLoader.dataset.vMean
    vStd = validLoader.dataset.vStd

    aMean = validLoader.dataset.aMean
    aStd = validLoader.dataset.aStd

    mean = torch.Tensor([vMean, aMean]).to(device)
    std = torch.Tensor([vStd, aStd]).to(device)

    losses = []
    vMapes = []
    vRmses = []
    vCorrs = []
    aCorrs = []

    with torch.no_grad():
        for i, (curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs) in enumerate(validLoader):

            curvatures = curvatures.to(device)
            targetSpeeds = targetSpeeds.to(device)
            histSpeeds = histSpeeds.to(device)
            targetAccelXs = targetAccelXs.to(device)
            histAccelXs = histAccelXs.to(device)

            decodeLength = targetSpeeds.size(1)-1

            curvatures = encoder(curvatures.unsqueeze(dim=1))

            predictions, alphas, alphas_target = decoder(curvatures, targetSpeeds[:, 0], histSpeeds, targetAccelXs, histAccelXs, decodeLength,
                                        validLoader.dataset.vMean, validLoader.dataset.vStd,
                                        validLoader.dataset.aMean, validLoader.dataset.aStd)

            targets = targetAccelXs[:, :-1].unsqueeze(-1)

            loss, metrics = criterion(predictions, targets, mean, std, alphas, alphas_target)

            aPreds = (predictions * aStd + aMean).squeeze(-1)
            curSpeed = targetSpeeds[:, 0]*vStd+vMean
            vPreds = torch.cumsum(aPreds, dim=1) + (curSpeed).unsqueeze(-1)
            vTrues = targetSpeeds[:, 1:]*vStd+vMean
            aTrues = targetAccelXs[:, :-1]*aStd+aMean

            vMape, vRmse = accuracy(vPreds, vTrues)
            vCorr, aCorr = pearsonr(torch.cat((vPreds.unsqueeze(-1), aPreds.unsqueeze(-1)), 2), torch.cat((vTrues.unsqueeze(-1), aTrues.unsqueeze(-1)), 2))

            losses.append(loss.item())
            vMapes.append(vMape)
            vRmses.append(vRmse)
            vCorrs.append(vCorr)
            aCorrs.append(aCorr)

    return np.mean(np.array(losses)), np.mean(np.array(vMapes)), np.mean(np.array(vRmses)), np.mean(np.array(vCorrs)), np.mean(np.array(aCorrs))

if __name__ == "__main__":
    main()
