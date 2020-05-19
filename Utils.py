
import pdb

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import random
import json
import h5py
from scipy.misc import imread, imresize

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
                           'sTarget': [,] ,
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

        # if os.path.isfile(imgPath):
        #     continue

        preview = dh.get_preview(i, 'TIME')
        x = preview['PreviewY']  # 차량의 진행 방향을 +y로 하려면, x, y 반전만 하면 되는게 맞는지?
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


class dataNormalization:
    def __init__(self, data):
        self.mu, self.std = norm.fit(data)
        self.data = (data-self.mu)/self.std

    def normalization(self, data):
        return (data-self.mu)/self.std

    def denormalization(self, data):
        return (data*self.std) + self.mu

def files2df(DATA_PATH, files):
    # This function originally aimed to read multiple data files into one data set,
    # but currently only supports one data file. (시간 순서 때문에 현재 함수로는 여러개의 파일을 합칠 수가 없음.)
    df_list = []
    for file in files:
        filepath = os.path.join(DATA_PATH, file)
        assert os.path.exists(filepath), 'No filename {}'.format(file)
        df_list.append(pd.read_csv(filepath))
    return pd.concat(df_list)

def draw_result_graph_fitting(true, predict, curvature, previewType, predictLength, true_curv=None, idxFrom=0, idxTo=None):
    if idxTo is None:
        idxTo = len(true)

    fig, ax = plt.subplots()
    line1 = ax.plot(true[idxFrom:idxTo], 'b.-', label='True', linewidth=2)
    line2 = ax.plot(predict[idxFrom:idxTo], 'r.-', label='Predict', linewidth=2)

    ax1 = ax.twinx()
    line3 = ax1.plot([(-1)*x for x in curvature[idxFrom:idxTo]], 'k--', label='Curvature', linewidth=1)

    if true_curv is not None:
        line4 = ax.plot(true_curv[idxFrom:idxTo], 'b--', label='True_curv', linewidth=1)
        lines = line1 + line2 + line3 + line4
    else:
        lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs)

    rmse = np.sqrt(sum((true[idxFrom:idxTo] - predict[idxFrom:idxTo])**2)/true[idxFrom:idxTo].shape[0])
    mape = 100*sum(abs(true[idxFrom:idxTo] - predict[idxFrom:idxTo])/true[idxFrom:idxTo])/true[idxFrom:idxTo].shape[0]
    print('RMSE:', rmse)
    print('MAPE:', mape)
    print(' ')

    if previewType == 'TIME':
        ax.set_title('{}s ahead (RMSE:{:.3f}, MAPE:{:.3f})'.format(predictLength, rmse, mape))
    elif previewType == 'DISTANCE':
        pass
    ax.set_xlabel('Time [0.1s]')
    ax.set_ylabel('GPS_Speed')
    ax1.set_ylabel('(-1)*|Curvature|')
    plt.show()

def draw_result_graph(true, predict):
    pass

def buildDataset4fit(df, previewHelper, previewType):
    ks = []
    for idx in range(len(df)):
        print('Building training set... {}/{}'.format(idx, len(df)), end='\r')

        preview = previewHelper.get_preview(idx, previewType)
        ks.append(preview['Curvature'])

    pdb.set_trace()
    pad = len(max(ks, key=len))  # just for saving data pair as .npy
    ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])

    return ks_arr

if __name__ == "__main__":
    '''
        previewData
            |__ drdName
                |__ previewImgs (.jpg)
            |__ dataset_drdName.json
    '''

    dataPath = './previewData'

    # drdFile = './Data/mkkim-recoder-scz_msgs.csv'
    # drdName = 'mkkim'  # recFreq = 10

    drdFile = './Data/std_001.csv'  # drd: driving record data
    drdName = 'std001'  # recFreq = 20

    recFreq = 20  # [Hz]
    dvRate = 2  # predict every (recFreq/sWindow)Hz record points [unit:(1/recFreq)]
    previewTime = 10  # [s]

    # Save previews as imgs and corresponding speed targets
    create_dataset(dataPath, drdFile, drdName, previewTime=previewTime, myDPI=120, recFreq=recFreq, dvRate=dvRate)

    # Create input files from dataset.json file for model training
    create_input_files(dataPath, drdName, previewTime=previewTime)
