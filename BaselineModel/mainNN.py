import pdb

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # for helper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import DataHelper
from Utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from tqdm import tqdm

''' ================ '''
''' = Path Setting = '''
''' ================ '''
DATA_PATH = '../DATA'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(current_dir, 'DATA')


''' ============= '''
''' = Data Load = '''
''' ============= '''
trteRate = 0.7  # % of data from the beginning according to time sequence
recFreq = 10  # 10Hz

datafiles_tr = ['mkkim-recoder-scz_msgs.csv']
df_tr_name = 'mkkim-{}%'.format(int(trteRate*100))  # just for caching data
df_tr = files2df(DATA_PATH, datafiles_tr)

datafiles_te = ['mkkim-recoder-scz_msgs.csv']
df_te_name = 'mkkim-{}%-te'.format(int(trteRate*100))
df_te = files2df(DATA_PATH, datafiles_te)

df_tr = df_tr.iloc[:int(len(df_tr)*trteRate)]
df_te = df_te.iloc[int(len(df_te)*trteRate):]


''' ==================== '''
''' = Build TR/TE data = '''
''' ==================== '''
previewHelper_tr = DataHelper(df_tr)
previewHelper_te = DataHelper(df_te)

previewType = 'TIME'

preview_time = 10
previewHelper_tr.set_preview_time(preview_time)
previewHelper_te.set_preview_time(preview_time)

predictLength = 5  # predict the speed at 5s ahead

# ks, vs, vsdiff
dataset4train = 'ks_{}_{}s.npy'.format(df_tr_name, preview_time)
if not os.path.isfile(dataset4train):
    ks = buildDataset4fit(df_tr, previewHelper_tr, previewType)
    np.save(dataset4train, ks)
else:
    ks = np.load(dataset4train)
vs = df_tr['GPS_Speed'].values
tr_true = vs[predictLength*recFreq:]  # target values
vsdiff = vs[predictLength*recFreq:] - vs[:-predictLength*recFreq]

vs = vs[:-predictLength*recFreq]  # X1
ks = ks[:-predictLength*recFreq, 1:]  # X2 (exclude curvature at every time index 't')
## record 데이터에서는 ks가 -1, -2 데이터가 nan임...
# previewType이 시간이고, data가 시간으로 정렬되어 있어도 preview curvature의 길이가 달라지나봄.

# 평균 preview 길이 구하기. (평균이라기보다 median)
avgLenPreview = int(np.median(np.sum(np.isnan(ks), axis=1)))
ks = ks[:, :-avgLenPreview]

notNaNidx = np.where(np.sum(np.isnan(ks), axis=1)==0)
ks_cut = abs(ks[notNaNidx])
vs_cut = vs[notNaNidx]
vsdiff_cut = vsdiff[notNaNidx]
tr_true = tr_true[notNaNidx]  # notNaNidx는 결국 순서만 맞으면 되므로

ks_norm = dataNormalization(ks_cut)
vs_norm = dataNormalization(vs_cut)
vsdiff_norm = dataNormalization(vsdiff_cut)

tr_x = np.hstack((np.expand_dims(vs_norm.data, 1), ks_norm.data))
tr_y = np.expand_dims(vsdiff_norm.data, 1)

print('Train data shape:', tr_x.shape)
print('Test  data shape:', tr_y.shape)


''' ============================== '''
''' = Build Model / Train & Test = '''
''' ============================== '''
tr_x = torch.from_numpy(tr_x)
tr_y = torch.from_numpy(tr_y)

model = nn.Sequential(
    nn.Linear(tr_x.shape[1], int(tr_x.shape[1]/2)),
    nn.LeakyReLU(0.2),
    nn.Linear(int(tr_x.shape[1]/2), 2),
    nn.LeakyReLU(0.2),
    nn.Linear(2, 1)
)

gpu = torch.device('cuda')
loss_func = nn.L1Loss().to(gpu)
# loss_func = nn.MSELoss().to(gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

model = model.to(gpu)
tr_x = tr_x.to(gpu)
tr_y = tr_y.to(gpu)

tr_x = tr_x.float()
tr_y = tr_y.float()

print('\n===========')
print('===Train===')
print('===========')
num_epoch = 50000*2
# num_epoch = 75000
# num_epoch = 5000
loss_array = []
for epoch in tqdm(range(num_epoch)):
# for epoch in range(num_epoch):
    optimizer.zero_grad()
    output = model(tr_x)

    loss = loss_func(output, tr_y)
    loss.backward()
    optimizer.step()

    loss_array.append(loss)

    # if epoch % 200 == 0:
        # print('epoch:', epoch, ' loss:', loss.item())

print('epoch:', epoch, ' loss:', loss.item())

plt.plot(loss_array)
plt.title('Loss')
plt.show()


tr_y = tr_y.cpu().detach().numpy()
output = output.cpu().detach().numpy()

tr_output = np.expand_dims(vs_cut, 1) + vsdiff_norm.denormalization(output)
rmse = np.sqrt(sum((tr_true - np.squeeze(tr_output))**2)/tr_true.shape[0])
mape = 100*sum(abs(tr_true - np.squeeze(tr_output))/tr_true)/tr_true.shape[0]
print('RMSE:', rmse)
print('MAPE:', mape)

## Model output (Trian) Graph
# plt.plot(tr_y, 'b', label='true')
# plt.plot(output, 'r', label='model output')
# plt.legend()
# plt.grid()
# plt.title('Model Output (normalized vdiff)')
# plt.show()
#
# ### 여기서 train의 predict랑 true랑 비교하는 graph 그리고,
# # tr_true
# tr_output = np.expand_dims(vs_cut, 1) + vsdiff_norm.denormalization(output)
#
# plt.plot(tr_true, 'b', label='True')
# plt.plot(tr_output, 'r', label='Model Output')
# plt.legend()
# plt.grid()
# plt.title('[Train] Model Output (speed prediction)')
# plt.show()

print('\n==========')
print('===Test===')
print('==========')
vs_te = df_te['GPS_Speed'].values
te_true = vs_te[predictLength*recFreq:]

vs_te = vs_te[:-predictLength*recFreq]

dataset4fit_te = 'ks_{}_{}s.npy'.format(df_te_name, preview_time)

if not os.path.isfile(dataset4fit_te):
    ks_te = buildDataset4fit(df_te, previewHelper_te, previewType)
    np.save(dataset4fit_te, ks_te)
else:
    ks_te = np.load(dataset4fit_te)

ks_te = ks_te[:-predictLength*recFreq, 1:]
ks_te = ks_te[:, :ks.shape[1]]

notNaNidx = np.where(np.sum(np.isnan(ks_te), axis=1)==0)
ks_te_cut = abs(ks_te[notNaNidx])
vs_te_cut = vs_te[notNaNidx]
te_true = te_true[notNaNidx]

ks_te_norm = ks_norm.normalization(ks_te_cut)
vs_te_norm = vs_norm.normalization(vs_te_cut)

te_x = np.hstack((np.expand_dims(vs_te_norm.data, 1), ks_te_norm.data))
te_x = torch.from_numpy(te_x)
te_x = te_x.to(gpu)
te_x = te_x.float()
output = model(te_x)
output = output.cpu().detach().numpy()

te_output = np.expand_dims(vs_te_cut, 1) + vsdiff_norm.denormalization(output)
rmse = np.sqrt(sum((te_true - np.squeeze(te_output))**2)/te_true.shape[0])
mape = 100*sum(abs(te_true - np.squeeze(te_output))/te_true)/te_true.shape[0]
print('RMSE:', rmse)
print('MAPE:', mape)

# plt.plot(te_true, 'b', label='True')
# plt.plot(vs_te_cut, 'b--', label='True Curv')
# plt.plot(te_output, 'r', label='Model Output')
# plt.legend()
# plt.grid()
#
# rmse = np.sqrt(sum((te_true - np.squeeze(te_output))**2)/te_true.shape[0])
# mape = 100*sum(abs(te_true - np.squeeze(te_output))/te_true)/te_true.shape[0]
#
# plt.xlabel('Time [0.1s]')
# plt.ylabel('GPS_Speed')
#
# plt.title('{}s ahead (Total, RMSE:{:.3f}, MAPE:{:.3f})'.format(predictLength, rmse, mape))
# plt.show()

# for a specific time period
sidx = 2500
fidx = 4500
plt.plot(te_true[sidx:fidx], 'b', label='True')
plt.plot(vs_te_cut[sidx:fidx], 'b--', label='True Curv')
plt.plot(te_output[sidx:fidx], 'r', label='Model Output')
plt.legend()
plt.grid()

rmse = np.sqrt(sum((te_true[sidx:fidx] - np.squeeze(te_output[sidx:fidx]))**2)/te_true[sidx:fidx].shape[0])
mape = 100*sum(abs(te_true[sidx:fidx] - np.squeeze(te_output[sidx:fidx]))/te_true[sidx:fidx])/te_true[sidx:fidx].shape[0]
print('RMSE:', rmse)
print('MAPE:', mape)

plt.xlabel('Time [0.1s]')
plt.ylabel('GPS_Speed')

plt.title('{}s ahead (Total, RMSE:{:.3f}, MAPE:{:.3f})'.format(predictLength, rmse, mape))
plt.show()
