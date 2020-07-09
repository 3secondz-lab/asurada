import pdb

import torch
import torch.optim
import torch.utils.data
from torch import nn

from model import Encoder, Decoder
from dataset import Dataset
from constants import device

from utils import *

''' DataSet1: IJF_D* (Real) '''
# driverNum = 4
# dataName = 'IJF_D{}_20_2_2_10_10_10'.format(driverNum)  # cWindow [s], vWindow [s], vpWindow [s], cUnit [Hz], vUnit [Hz], vpUnit [Hz]

''' Dataset2: 003_0422_IJ (Real)'''
# dataName = 'IJ_20_2_2_10_10_10'

''' Dataset3: YJ (Simulated) '''
dataName = 'YJ_TD_100_2_2_10_10_10'

''' Model '''
chpt_enc_path = None
chpt_dec_path = None
chpt_stat_path = None
# chpt_enc_path = './BEST_checkpoint_ENC_{}.pth.tar'.format(dataName)
# chpt_dec_path = './BEST_checkpoint_DEC_{}.pth.tar'.format(dataName)
# chpt_stat_path ='./BEST_stat_{}.pickle'.format(dataName)

## Model parameters
input_size = 1
hidden_size = 64
output_size = 1

## Training parameters
start_epoch = 0
epochs = 200
batch_size = 256
workers = 0  # 0 for windows
encoder_lr = 1e-4
decoder_lr = 1e-4
best_mape = 100.
best_rmse = 20.

loss_tr = AverageMeter()
mape_tr = AverageMeter()
rmse_tr = AverageMeter()

mape_tr_1s = AverageMeter()
rmse_tr_1s = AverageMeter()

loss_vl = AverageMeter()
mape_vl = AverageMeter()
rmse_vl = AverageMeter()

mape_vl_1s = AverageMeter()
rmse_vl_1s = AverageMeter()

def main():

    global start_epoch, best_mape, best_rmse

    encoder_c = Encoder(input_size=input_size, enc_dim=hidden_size)
    encoder_d = Encoder(input_size=input_size, enc_dim=hidden_size)
    decoder = Decoder(enc_dim=hidden_size, dec_dim=hidden_size, att_dim=hidden_size, output_dim=output_size)

    encoder_c_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_c.parameters()),
                                         lr=encoder_lr)
    encoder_d_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_d.parameters()),
                                         lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)

    # if chpt_enc_path is not None:
    #     encoder.load_state_dict(torch.load(chpt_enc_path))
    #     decoder.load_state_dict(torch.load(chpt_dec_path))
    #
    #     with open(chpt_stat_path, 'rb') as f:
    #         chpt_stat = pickle.load(f)
    #
    #     start_epoch = chpt_stat['epoch'] + 1  # restart at chpt_stat['epoch'] + 1
    #     loss_tr_list = chpt_stat['loss_tr_list']
    #     loss_vl_list = chpt_stat['loss_vl_list']
    #
    #     mape_tr_list = chpt_stat['mape_tr_list']
    #     mape_vl_list = chpt_stat['mape_vl_list']
    #
    #     rmse_tr_list = chpt_stat['rmse_tr_list']
    #     rmse_vl_list = chpt_stat['rmse_vl_list']
    #
    #     print('Checkpoint loaded:\nNext epoch: {}\nCurrent list len (start at 0): {}'.format(start_epoch, len(loss_tr_list)))

    encoder_c = encoder_c.to(device)
    encoder_d = encoder_d.to(device)
    decoder = decoder.to(device)

    criterion = nn.MSELoss().to(device)

    trainLoader = torch.utils.data.DataLoader(Dataset(dataName, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    cMean_tr = trainLoader.dataset.cMean
    cStd_tr = trainLoader.dataset.cStd
    sMean_tr = trainLoader.dataset.sMean
    sStd_tr = trainLoader.dataset.sStd

    validLoader = torch.utils.data.DataLoader(Dataset(dataName, 'TEST',
        cMean=cMean_tr, cStd=cStd_tr, sMean=sMean_tr, sStd=sStd_tr),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    if chpt_enc_path is None:
        loss_tr_list = []
        loss_vl_list = []

        mape_tr_list = []
        mape_vl_list = []

        rmse_tr_list = []
        rmse_vl_list = []

    for epoch in range(start_epoch, epochs):
        train(trainLoader=trainLoader,
              encoder_c=encoder_c,
              encoder_d=encoder_d,
              decoder=decoder,
              criterion=criterion,
              encoder_c_optimizer=encoder_c_optimizer,
              encoder_d_optimizer=encoder_d_optimizer,
              decoder_optimizer=decoder_optimizer)


        recent_mape, recent_rmse = validate(validLoader=validLoader,
                                            encoder_c=encoder_c,
                                            encoder_d=encoder_d,
                                            decoder=decoder,
                                            criterion=criterion)

        print("""Epoch: {0}\t
                 Loss: {loss.val:.4f} ({loss.avg:.4f})\t
                 MAPE: {mape.val:.4f} ({mape.avg:.4f})\t
                 RMSE: {rmse.val:.4f} ({rmse.avg:.4f})\t

                 VAL Loss: {loss_vl.val:.4f} ({loss_vl.avg:.4f})\t
                 VAL MAPE: {mape_vl.val:.4f} ({mape_vl.avg:.4f})\t
                 VAL RMSE: {rmse_vl.val:.4f} ({rmse_vl.avg:.4f})\t""".format(epoch, loss=loss_tr, mape=mape_tr, rmse=rmse_tr,
                                                                                    loss_vl=loss_vl, mape_vl=mape_vl, rmse_vl=rmse_vl))

        print('')

        print("""MAPE1s: {mape.val:.4f} ({mape.avg:.4f})\tRMSE1s: {rmse.val:.4f} ({rmse.avg:.4f})""".format(mape=mape_tr_1s, rmse=rmse_tr_1s))
        print("""MAPE1s: {mape.val:.4f} ({mape.avg:.4f})\tRMSE1s: {rmse.val:.4f} ({rmse.avg:.4f})\n""".format(mape=mape_vl_1s, rmse=rmse_vl_1s))

        loss_tr_list.append(loss_tr.val)
        loss_vl_list.append(loss_vl.val)

        mape_tr_list.append(mape_tr.val)
        mape_vl_list.append(mape_vl.val)

        rmse_tr_list.append(rmse_tr.val)
        rmse_vl_list.append(rmse_vl.val)

        is_best_mape = recent_mape < best_mape
        best_mape = min(recent_mape, best_mape)

        is_best_rmse = recent_rmse < best_rmse
        best_rmse = min(recent_rmse, best_rmse)

        save_checkpoint(dataName, epoch, encoder_c, encoder_d, decoder, cMean_tr, cStd_tr, sMean_tr, sStd_tr, is_best_mape,
                        loss_tr_list, loss_vl_list, mape_tr_list, mape_vl_list, rmse_tr_list, rmse_vl_list)

        save_checkpoint(dataName, epoch, encoder_c, encoder_d, decoder, cMean_tr, cStd_tr, sMean_tr, sStd_tr, is_best_mape,
                        loss_tr_list, loss_vl_list, mape_tr_list, mape_vl_list, rmse_tr_list, rmse_vl_list, RMSE=True)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(loss_tr_list)
    plt.plot(loss_vl_list, 'r')

    plt.figure()
    plt.plot(mape_tr_list)
    plt.plot(mape_vl_list, 'r')

    plt.figure()
    plt.plot(rmse_tr_list)
    plt.plot(rmse_vl_list, 'r')

    plt.show()

def train(trainLoader, encoder_c, encoder_d, decoder, criterion, encoder_c_optimizer, encoder_d_optimizer, decoder_optimizer):

    encoder_c.train()
    encoder_d.train()
    decoder.train()

    mean = trainLoader.dataset.sMean
    std = trainLoader.dataset.sStd

    for i, (curvatures, targetSpeeds, histSpeeds) in enumerate(trainLoader):

        curvatures = curvatures.to(device)
        targetSpeeds = targetSpeeds.to(device)
        histSpeeds = histSpeeds.to(device)

        decodeLength = targetSpeeds.size(1)-1

        enc_hiddens_v = encoder_d(histSpeeds)
        enc_hiddens_c = encoder_c(curvatures)

        predictions, alphas_d, alphas_c = decoder(enc_hiddens_v, enc_hiddens_c, targetSpeeds[:, 0], decodeLength)

        targets = targetSpeeds[:, 1:]

        loss = criterion(predictions, targets)

        encoder_c_optimizer.zero_grad()
        encoder_d_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()

        encoder_c_optimizer.step()
        encoder_d_optimizer.step()
        decoder_optimizer.step()

        mape, rmse = accuracy(predictions*std+mean, targets*std+mean)

        loss_tr.update(loss.item())
        mape_tr.update(mape)
        rmse_tr.update(rmse)

        mape_1s, rmse_1s = accuracy(predictions[:, 9]*std+mean, targets[:, 9]*std+mean)  # 1s: 10개 앞.
        mape_tr_1s.update(mape_1s)
        rmse_tr_1s.update(rmse_1s)

def validate(validLoader, encoder_c, encoder_d, decoder, criterion):
    encoder_c.eval()
    encoder_d.eval()
    decoder.eval()

    mean = validLoader.dataset.sMean
    std = validLoader.dataset.sStd
    with torch.no_grad():
        for i, (curvatures, targetSpeeds, histSpeeds) in enumerate(validLoader):
            curvatures = curvatures.to(device)
            targetSpeeds = targetSpeeds.to(device)
            histSpeeds = histSpeeds.to(device)

            decodeLength = targetSpeeds.size(1)-1

            enc_hiddens_v = encoder_d(histSpeeds)
            enc_hiddens_c = encoder_c(curvatures)
            predictions, alphas_d, alphas_c = decoder(enc_hiddens_v, enc_hiddens_c, targetSpeeds[:, 0], decodeLength)

            targets = targetSpeeds[:, 1:]

            loss = criterion(predictions, targets)

            mape, rmse = accuracy(predictions*std+mean, targets*std+mean)

            loss_vl.update(loss)
            mape_vl.update(mape)
            rmse_vl.update(rmse)

            mape_1s, rmse_1s = accuracy(predictions[:, 9]*std+mean, targets[:, 9]*std+mean)
            mape_vl_1s.update(mape_1s)
            rmse_vl_1s.update(rmse_1s)

    return mape, rmse

if __name__ == "__main__":
    main()
