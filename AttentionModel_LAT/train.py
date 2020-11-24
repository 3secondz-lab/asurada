import pdb

import torch
import torch.optim
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model_G_LA import Encoder, DecoderWithAttention  # lateral offset만으로는 local attention model에 적용 가능함.
# vNumber = 1  # hiddenDimension = 16
vNumber = 2  # hiddenDimension = 32

from dataset import Dataset
from constants import device

from utils import *
import matplotlib.pyplot as plt
import scipy
import shutil
from random import random

''' ======================================================================== '''
''' Data '''
driver = 'YJ'  # mugello, spa, imola, magione
circuit_tr = ['mugello', 'magione', 'imola']
circuit_vl = ['spa']  # test임.

curvatureLength = 250  # unit [m]
historyLength = 10  # unit [0.1s]
predLength = 20  # unit [0.1s]

training_gt_rate = 0.5  # 0: 예측값으로만 학습
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

''' Model '''
hiddenDimension = 32
encoder_dim = hiddenDimension
lstm_input_dim = historyLength + 1  # history - vt
decoder_dim = hiddenDimension  # hidden state
attention_dim = hiddenDimension
output_dim = 1  # diff(v)만!
criterion = nn.MSELoss().to(device)

''' Training parameters '''
start_epoch = 0
epochs = 150  # about 2Hr.
batch_size = 1024
workers = 0  # 0 for windows
encoder_lr = 1e-3
decoder_lr = 1e-3

epochs_since_improvement = 0
best_loss_tr = 10e5

grad_clip = 5.


def main():
    global epochs_since_improvement, best_loss_tr

    encoder = Encoder(hiddenDimension)
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
    lMean_tr = trainLoader.dataset.lMean
    lStd_tr = trainLoader.dataset.lStd

    validLoader = torch.utils.data.DataLoader(Dataset(driver, circuit_vl,
                                            curvatureLength, historyLength, predLength,
                                            cMean=cMean_tr, cStd=cStd_tr, vMean=vMean_tr, vStd=vStd_tr, aMean=aMean_tr, aStd=aStd_tr,
                                            lMean=lMean_tr, lStd=lStd_tr),
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('Training version.{} (A->V)'.format(vNumber))
    print('Training data ({} - {})'.format(driver, circuit_tr))
    print('Validation data ({} - {})'.format(driver, circuit_vl))
    print('curvature len {}'.format(curvatureLength))
    print('history len {}'.format(historyLength))

    print('\nTraining...\n')

    # for epoch in tqdm(range(start_epoch, epochs)):
    print('Epoch\tLoss\tlMape\tlRmse')
    for epoch in range(start_epoch, epochs):

        loss, lMape, lRmse = train(trainLoader=trainLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            encoder_optimizer=encoder_optimizer,
                            decoder_optimizer=decoder_optimizer,
                            epoch=epoch)
        print('{}\t{}\t{}\t{}'.format(epoch, loss, lMape, lRmse))

        writer.add_scalars('Loss', {'tr':loss}, epoch)
        writer.add_scalars('MAPE', {'tr':lMape}, epoch)  # Lateral offset
        writer.add_scalars('RMSE', {'tr':lRmse}, epoch)  # lateral offset

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
            loss_vl, lMape_vl, lRmse_vl = validate(validLoader=validLoader,
                                         encoder=encoder,
                                         decoder=decoder,
                                         criterion=criterion)
            print('{}\t{}\t{}\t{} (Valid)'.format(epoch, loss_vl, lMape_vl, lRmse_vl))

            writer.add_scalars('Loss', {'vl':loss_vl}, epoch)
            writer.add_scalars('MAPE', {'vl':lMape_vl}, epoch)
            writer.add_scalars('RMSE', {'vl':lRmse_vl}, epoch)

        if epoch%10 == 0:
            save_checkpoint(chptFolderPath, encoder, decoder, epoch, cMean_tr, cStd_tr, vMean_tr, vStd_tr, aMean_tr, aStd_tr,  lMean_tr, lStd_tr, curvatureLength, historyLength)
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

    lMean = trainLoader.dataset.lMean
    lStd = trainLoader.dataset.lStd

    mean = torch.Tensor([vMean, aMean]).to(device)
    std = torch.Tensor([vStd, aStd]).to(device)

    losses = []
    lMapes = []
    lRmses = []

    for i, (curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs, targetOffsets, histOffsets) in enumerate(trainLoader):

        curvatures = curvatures.to(device)  # [0]: 1m ahead
        targetSpeeds = targetSpeeds.to(device)  # [0]: currentSpeed, v_t
        histSpeeds = histSpeeds.to(device)
        targetAccelXs = targetAccelXs.to(device)  # [0]: a_t 부터 예측 target임.
        histAccelXs = histAccelXs.to(device)
        targetOffsets = targetOffsets.to(device) #''' 추가 '''
        histOffsets = histOffsets.to(device)

        decodeLength = targetSpeeds.size(1)-1  # [0]: currentSpeed

        curvatures = encoder(curvatures.unsqueeze(dim=1))

        if random() >= training_gt_rate:
            # predictions, alphas = decoder(curvatures, targetSpeeds[:, 0], histSpeeds, targetAccelXs, histAccelXs, decodeLength,
            #                             trainLoader.dataset.vMean, trainLoader.dataset.vStd,
            #                             trainLoader.dataset.aMean, trainLoader.dataset.aStd)  # 기존 코드
            predictions, alphas = decoder(curvatures, targetOffsets[:, 0], histOffsets, targetAccelXs, histAccelXs, decodeLength,
                                        trainLoader.dataset.vMean, trainLoader.dataset.vStd,
                                        trainLoader.dataset.aMean, trainLoader.dataset.aStd)  # 사실상 curve, offset, decodeLength만 필요
        else:
            # predictions, alphas = decoder(curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs, decodeLength,
            #                             trainLoader.dataset.vMean, trainLoader.dataset.vStd,
            #                             trainLoader.dataset.aMean, trainLoader.dataset.aStd)  # 기존 코드
            predictions, alphas = decoder(curvatures, targetOffsets, histOffsets, targetAccelXs, histAccelXs, decodeLength,
                                        trainLoader.dataset.vMean, trainLoader.dataset.vStd,
                                        trainLoader.dataset.aMean, trainLoader.dataset.aStd)


        targets = targetOffsets[:, 1:].unsqueeze(-1)

        loss = criterion(predictions, targets)

        stepIdx += 1
        writer.add_scalars('MSE', {'tr':loss.item()}, stepIdx)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            clip_gradient(encoder_optimizer, grad_clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        lPreds = predictions * lStd + lMean
        lTrues = targetOffsets[:, 1:]*lStd+lMean

        lMape = torch.mean(torch.abs(lPreds-lTrues.unsqueeze(-1))/abs(lTrues.unsqueeze(-1))*100)
        lRmse = torch.sqrt(torch.mean((lPreds-lTrues.unsqueeze(-1))**2))

        losses.append(loss.item())

        lMapes.append(lMape.item())
        lRmses.append(lRmse.item())

    return np.mean(np.array(losses)), np.mean(np.array(lMapes)), np.mean(np.array(lRmses))

def validate(validLoader, encoder, decoder, criterion):  # 여기는 무조건 예측값으로 lstm 계속 진행
    encoder.eval()
    decoder.eval()

    vMean = validLoader.dataset.vMean
    vStd = validLoader.dataset.vStd

    aMean = validLoader.dataset.aMean
    aStd = validLoader.dataset.aStd

    lMean = validLoader.dataset.lMean
    lStd = validLoader.dataset.lStd

    mean = torch.Tensor([vMean, aMean]).to(device)
    std = torch.Tensor([vStd, aStd]).to(device)

    losses = []
    lMapes = []
    lRmses = []
    with torch.no_grad():
        for i, (curvatures, targetSpeeds, histSpeeds, targetAccelXs, histAccelXs, targetOffsets, histOffsets) in enumerate(validLoader):


            curvatures = curvatures.to(device)
            targetSpeeds = targetSpeeds.to(device)
            histSpeeds = histSpeeds.to(device)
            targetAccelXs = targetAccelXs.to(device)
            histAccelXs = histAccelXs.to(device)
            targetOffsets = targetOffsets.to(device)
            histOffsets = histOffsets.to(device)

            decodeLength = targetSpeeds.size(1)-1

            curvatures = encoder(curvatures.unsqueeze(dim=1))
            predictions, alphas = decoder(curvatures, targetOffsets[:, 0], histOffsets, targetAccelXs, histAccelXs, decodeLength,
                                        validLoader.dataset.vMean, validLoader.dataset.vStd,
                                        validLoader.dataset.aMean, validLoader.dataset.aStd)  # 사실상 curve, offset, decodeLength만 필요

            targets = targetOffsets[:, 1:].unsqueeze(-1)

            loss = criterion(predictions, targets)

            lPreds = predictions * lStd + lMean
            lTrues = targetOffsets[:, 1:]*lStd+lMean

            lMape = torch.mean(torch.abs(lPreds-lTrues.unsqueeze(-1))/abs(lTrues.unsqueeze(-1))*100)
            lRmse = torch.sqrt(torch.mean((lPreds-lTrues.unsqueeze(-1))**2))

            losses.append(loss.item())
            lMapes.append(lMape.item())
            lRmses.append(lRmse.item())

    return np.mean(np.array(losses)), np.mean(np.array(lMapes)), np.mean(np.array(lRmses))

if __name__ == "__main__":
    main()
