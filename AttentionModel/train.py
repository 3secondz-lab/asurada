import pdb

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from ModelAtt import Encoder, DecoderWithAttention
from datasets import *
from UtilsAtt import *


# Data parameters
data_folder = './previewData'  # folder with data files saved by create_input_files.py
data_name = 'std001_10_previewTime_0.5_sWindow'  # base name shared by data files

# Model parameters
attention_dim = 64  # 512 -> 256 -> 128 -> 64 -> 32
decoder_dim = 64
dropout = 0.5  # 0.5 -> 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation MAPE
batch_size = 128  # 32 -> 64
workers = 0  # # for data-loading with h5py !!!!! for window only 0 !!!!!
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 1e-4  # learning rate for decoder (4e-4 -> 1e-4)
# encoder_lr = 5e-5  # learning rate for encoder if fine-tuning
# decoder_lr = 5e-5  # learning rate for decoder (4e-4 -> 1e-4)
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_mape = 100.  # early stopping criteria
print_freq = 30  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    '''
        Training and Validation
    '''

    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, dataName, best_mape

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       decoder_dim=decoder_dim,
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # normalize the image by the mean and std of the ImageNet images' RGB channels.
    train_loader = torch.utils.data.DataLoader(
        SpeedDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        SpeedDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]),
        mean=train_loader.dataset.mean, std=train_loader.dataset.std),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:  # 8 -> 3
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_mape = validate(val_loader=val_loader,
                               encoder=encoder,
                               decoder=decoder,
                               criterion=criterion)

        # Check if there was an improvement
        is_best = recent_mape < best_mape
        best_mape = min(recent_mape, best_mape)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        # train_loader: for trMean, trStd
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, train_loader, recent_mape, is_best)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  ##
    mapes = AverageMeter()
    rmses = AverageMeter()

    start = time.time()

    # Batches
    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    for i, (imgs, target_speeds) in enumerate(train_loader):
        ''' imgs: (batch_size, 3, 224, 224)
            target_speeds: (batch_size, 21)
        '''
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        target_speeds = target_speeds.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        predictions, alphas = decoder(imgs, target_speeds)

        # # Since we decoded starting with current_speed, the targets are all predicted speeds after current speed
        targets = target_speeds[:, 1:]  # exclude current speed

        # Calculate loss
        loss = criterion(predictions, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        mape, rmse = accuracy(predictions*std+mean, targets*std+mean)
        batch_time.update(time.time() - start)
        losses.update(loss.item())
        mapes.update(mape)
        rmses.update(rmse)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print("""Epoch: [{0}][{1}/{2}]\t
                  Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t
                  Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t
                  Loss {loss.val:.4f} ({loss.avg:.4f})\t
                  MAPE {mape.val:.4f} ({mape.avg:.4f})\t
                  RMSE {rmse.val:.4f} ({rmse.avg:.4f})""".format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          mape=mapes, rmse=rmses))

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    losses = []
    mapes = []
    rmses = []

    mean = val_loader.dataset.mean  # == train_loader.dataset.mean
    std = val_loader.dataset.std  # == train_loader.dataset.std
    with torch.no_grad():

        # Batches
        for i, (imgs, target_speeds) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            target_speeds = target_speeds.to(device)

            # Encode
            encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
            batch_size = encoder_out.size(0)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            # Flatten encoding
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            current_speed = target_speeds[:, 0].reshape(-1, 1)
            current_speed = (current_speed - mean) / std
            current_speed = current_speed.to(device)

            decode_length = target_speeds.size(1) - 1

            predictions = torch.zeros(batch_size, decode_length).to(device)
            alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)

            h, c = decoder.init_hidden_state(encoder_out)

            for t in range(decode_length):

                attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

                if t == 0:
                    h, c = decoder.decode_step(torch.cat([current_speed, attention_weighted_encoding], dim=1), (h, c))
                else:
                    h, c = decoder.decode_step(torch.cat([preds, attention_weighted_encoding], dim=1), (h, c))

                preds = decoder.fc(h)

                predictions[:, t] = preds.squeeze(dim=1)
                alphas[:, t, :] = alpha

            targets = target_speeds[:, 1:]

            # Calculate loss
            loss = criterion(predictions, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            mape, rmse = accuracy(predictions*std+mean, targets*std+mean)

            losses.append(loss)
            mapes.append(mape)
            rmses.append(rmse)

        print(' ')

        avgLoss = sum(losses)/len(losses)
        avgMAPE = sum(mapes)/len(mapes)
        avgRMSE = sum(rmses)/len(rmses)
        print('VAL * LOSS - {:.4f} MAPE - {:.4f} RMSE - {:.4f}'.format(avgLoss, avgMAPE, avgRMSE))

    return avgMAPE

if __name__ == '__main__':
    main()