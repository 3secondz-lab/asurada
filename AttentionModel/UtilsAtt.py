import pdb

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import random
import json
import h5py
from scipy.misc import imread, imresize
import torch

from helper import DataHelper


def preview2img(x, y):
    '''
        Converts and returns the preview's x and y information to an image.
        [ref] https://github.com/matplotlib/matplotlib/issues/7940/
    '''
    fig, ax = plt.subplots(figsize=(4,4))  # 매 preview 마다 x, y 범위가 다른데, 이거 처리 어떻게?
    ax.axis = ('off')
    ax.plot(x, y, color='k', linewidth=4)
    ax.plot(x[0], y[0], color='k', marker='o', markersize=10)  # starting point (0, 0)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    return fig, extent

def create_dataset(dataPath, drdFile, drdName, previewTime, myDPI, recFreq=10, dvRate=2, trvRate=0.8):
    '''
        Create a dataset for Training/Validation
        # input:
        # output: save 'dataSet' in './dataPath/dataset_drdName.json'
               dataset = [{'drdName': ,
                           'imgFName': ,
                           'sTarget': [current_speed(s0), target_speeds(s1:sT)],
                           'sWindow': ,
                           'split': },]
    '''
    imgDir = '{}/{}'.format(dataPath, drdName)  # directory for saving preview imgs
    if not os.path.isdir(imgDir):
        os.mkdir(imgDir)

    df = pd.read_csv(drdFile)

    dh = DataHelper(df)
    dh.set_preview_time(previewTime)

    dataSet = []
    s = df['GPS_Speed'].values
    sWindow = int(recFreq/dvRate)  # predict every sHz record points [unit:(1/recFreq)]
    for i in tqdm(range(len(df))):
        data = {}

        imgFName = '{:06}.jpg'.format(i)
        imgPath = os.path.join(imgDir, imgFName)

        preview = dh.get_preview(i, 'TIME')
        x = preview['PreviewY']
        y = preview['PreviewX']

        sTarget = s[i:i+(previewTime*recFreq)+1]

        if len(sTarget) <= previewTime*recFreq:
            continue  # except when the previwe is not long enough

        sTarget = sTarget[range(0, len(sTarget), sWindow)]

        # imgs
        fig, extent = preview2img(x, y)
        fig.savefig(imgPath, dpi=myDPI, bbox_inches=extent, format='jpg',
                                facecolor=fig.get_facecolor(), transparent=True)
        plt.close(fig)

        data['drdName'] = drdName
        data['imgFName'] = imgFName
        data['sTarget'] = sTarget.tolist()
        data['sWindow'] = sWindow * (1/recFreq)
        data['split'] = 'val' if random() > trvRate else 'train'
        data['previewTime'] = previewTime

        dataSet.append(data)

    with open('{}/dataset_{}.json'.format(dataPath, drdName), 'w') as j:
        json.dump(dataSet, j)

def create_input_files(dataPath, drdName, previewTime):
    '''
        Create input files from a json file
    '''
    jsonPath = '{}/dataset_{}.json'.format(dataPath, drdName)
    assert os.path.isfile(jsonPath), 'Run create_dataset() first.'

    # Read JSON
    with open(jsonPath, 'r') as j:
        data = json.load(j)

    # Read image paths and target speed for each image
    tr_ImgPaths = []
    tr_ImgSpeeds = []
    val_ImgPaths = []
    val_ImgSpeeds = []

    for img in data:
        imgPath = os.path.join(dataPath, img['drdName'], img['imgFName'])

        if img['split'] in {'train'}:
            tr_ImgPaths.append(imgPath)
            tr_ImgSpeeds.append(img['sTarget'])
        elif img['split'] in {'val'}:
            val_ImgPaths.append(imgPath)
            val_ImgSpeeds.append(img['sTarget'])

    # Sanity check
    assert len(tr_ImgPaths) == len(tr_ImgSpeeds)
    assert len(val_ImgPaths) == len(val_ImgSpeeds)

    # Create a base name for all output files
    baseFName = drdName + '_{}_previewTime_{}_sWindow'.format(img['previewTime'], img['sWindow'])

    for imgPaths, imgSpeeds, split in [(tr_ImgPaths, tr_ImgSpeeds, 'TRAIN'),
                                       (val_ImgPaths, val_ImgSpeeds, 'VAL')]:

        with h5py.File(os.path.join(dataPath, split + '_IMAGES_' + baseFName + '.hdf5'), 'w') as h:
            imgs = h.create_dataset('images', (len(imgPaths), 3, 224, 224), dtype = 'uint8')

            print("\nReading %s images and target Speeds, storing to file...\n" % split)

            speeds = []

            for i, path in enumerate(tqdm(imgPaths)):
                # Read images
                img = imread(imgPaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (224, 224))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                imgs[i] = img

                # Save target speed to json file
                speeds.append(imgSpeeds[i])

            assert imgs.shape[0] == len(speeds)

            with open(os.path.join(dataPath, split + '_SPEEDS_' + baseFName + '.json'), 'w') as j:
                json.dump(speeds, j)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(predict, true):
    mape = torch.mean(torch.abs(predict-true)/true*100)
    rmse = torch.sqrt(torch.mean((predict-true)**2))
    return mape.item(), rmse.item()

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    train_loader, mape, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param train_loader: for the mean/std values of training data
    :param mape: validation MAPE score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'mape': mape,
             'trMean': train_loader.dataset.mean,
             'trStd': train_loader.dataset.std,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
