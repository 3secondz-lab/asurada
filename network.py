import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
import os
from torch.autograd import Variable

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

class NetModel(nn.Module):
    def __init__(self, input, h1, h2, output, lr=0.1):
        super(NetModel, self).__init__()

        self.input = input
        self.h1 = h1
        self.h2 = h2
        self.output = output
        self.lr = lr

        self.linear1 = nn.Linear(input, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, output)


        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()


    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x
#
#
# class Agent:
#     def __init__(self, input_size, h1, h2, output_size, lr, batch_size):
#         self.input_size = input_size
#         self.output_size= output_size
#         self.lr = lr
#         self.batch_size = batch_size
#         self.predictor = NetModel(input_size, h1, h2, output_size, lr)
#
#     def update(self, input, output):
#         input = torch.Tensor(input)
#         output = torch.Tensor(output)
#         # epochs = 2*int(len(input)/batch_size)
#         loss_avg = 0
#         # for _ in range(epochs):
#         count = 1
#         past_loss = np.inf
#         while True:
#
#             batch_idx = np.random.choice(len(input), self.batch_size, replace=False)
#             batch_input = input[batch_idx]
#             batch_output = output[batch_idx]
#
#             # take gradient step
#             prediction = self.predictor(input)
#             loss = self.predictor.loss(prediction, output)
#             self.predictor.optimizer.zero_grad()
#             loss.mean().backward()
#             self.predictor.optimizer.step()
#
#             if count%100==0:
#                 self.save(count)
#             if self.predictor.lr > 10**(-4):
#                 if (abs(past_loss-loss.item())<0.005 or past_loss*1.05<loss.item()):
#                     self.predictor.lr *=0.95
#
#             print(self.predictor.lr)
#             past_loss = loss.item()
#             count +=1
#             print("{} loss {}".format(count, loss.item()))
#             if loss.item()<0.1:
#                 break
#         return
#
#     def save(self, filename):
#         torch.save(self.predictor.state_dict(), 'lr_modified_95/{}.pth'.format(filename))
#
#     def load(self, filename):
#         # torch.load_state_dict(self.predictor.state_dict(), '{}.pth'.format(filename))
#         self.predictor.load_state_dict(torch.load("learned/{}.pth".format(filename)))

class Agent:
    def __init__(self, df, df_name, previewHelper, previewType, \
                 input_size, h1, h2, output_size, lr, batch_size,\
                 use_throttle=False, use_steer_spd=False):

        self.previewHelper = previewHelper
        self.previewType = previewType
        self.preview_time = previewHelper.preview_time
        self.preview_distance = previewHelper.preview_distance


        self.input_size = input_size
        self.output_size= output_size
        self.lr = lr
        self.batch_size = batch_size

        self.use_throttle = use_throttle
        self.use_steer_spd = use_steer_spd

        self.rescale_velocity = 10 ** (-6)
        self.rescale_throttle = 10 ** (-6)
        self.rescale_steer_spd = 10 ** (-3)

        self.predictor = NetModel(input_size+use_throttle+use_steer_spd+1, h1, h2, output_size*(1+use_steer_spd), lr)
        # Data Loading (k vs. v)
        if previewType == 'TIME':
            self.dataset4fit = 'ks_{}_{}s.npy'.format(df_name, self.preview_time)
        elif previewType == 'DISTANCE':
            self.dataset4fit = 'ks_{}_{}m.npy'.format(df_name, self.preview_distance)


    def preprocess(self, tr_set):
        # if not os.path.isfile(self.dataset4fit):
        ks = []
        vs = []
        throttles = []
        steer_spds = []
        for idx in range(len(tr_set)):
            preview = self.previewHelper.get_preview(idx, self.previewType)
            ks.append(preview['Curvature'])
            vs.append(preview['GPS_Speed'])
            throttles.append(preview['ECU_THROTTLE'])
            steer_spds.append(preview['ECU_STEER_SPD'])
        pad = len(max(ks, key=len))
        ks_arr = np.array([k.tolist() + [np.nan]*(pad-len(k)) for k in ks])
        vs_arr = np.array([v.tolist() + [np.nan]*(pad-len(v)) for v in vs])
        np.save(self.dataset4fit, ks_arr)
        np.save('vs'+self.dataset4fit[2:], vs_arr)
        if self.use_throttle:
            throttles_arr = np.array([throttle.tolist() + [np.nan]*(pad-len(throttle)) for throttle in throttles])
            # np.save(~~~)
        if self.use_steer_spd:
            steer_spds_arr = np.array([steer_spd.tolist() + [np.nan]*(pad-len(steer_spd)) for steer_spd in steer_spds])
            # np.save(~~~)




        ks = np.load(self.dataset4fit)
        vs = np.load('vs'+self.dataset4fit[2:])
        if self.use_throttle:
            # throttles = np.load(~~)
            pass
        if self.use_steer_spd:
            # steer_spds = np.load(~~)
            pass

        # remove nan at the end of the lap
        ks = ks[:-100]
        vs = vs[:-100]
        if self.use_throttle:
            throttles = np.array(throttles[:-100])
        if self.use_steer_spd:
            steer_spds = np.array(steer_spds[:-100])

        # input and output sizes should be less than 101
        x_mem = ks[:, :self.input_size]
        y_mem = vs[:, :self.output_size]

        # concatenate the present velocity, throttle, or steering speed to curvature features if needed
        velocity = (y_mem[:,0] * self.rescale_velocity).reshape(len(x_mem),-1)
        x_mem = np.concatenate((velocity, x_mem), axis=1)

        if self.use_throttle:
            x_mem = np.concatenate((throttles[:,0].reshape(len(x_mem),-1) * self.rescale_throttle, x_mem), axis=1)
        if self.use_steer_spd:
            steer_spd_input = steer_spds[:,0].reshape(len(x_mem),-1) * self.rescale_steer_spd
            x_mem = np.concatenate((steer_spd_input, x_mem), axis=1)
            steer_spd_output = steer_spds[:,:self.output_size]
            y_mem = np.concatenate((steer_spd_output, y_mem), axis=1)

        return x_mem, y_mem

    def update(self, input, output):
        input = torch.Tensor(input)
        output = torch.Tensor(output)
        # epochs = 2*int(len(input)/batch_size)
        loss_avg = 0
        # for _ in range(epochs):
        count = 1
        past_loss = np.inf
        while True:
            batch_idx = np.random.choice(len(input), self.batch_size, replace=False)
            batch_input = input[batch_idx]
            batch_output = output[batch_idx]

            # take gradient step
            prediction = self.predictor(input)
            loss = self.predictor.loss(prediction, output)
            self.predictor.optimizer.zero_grad()
            loss.mean().backward()
            self.predictor.optimizer.step()

            if count%100==0:
                self.save(count)
            if self.predictor.lr > 10**(-4):
                if (abs(past_loss-loss.item())<0.005 or past_loss*1.05<loss.item()):
                    self.predictor.lr *=0.95

            print(self.predictor.lr)
            past_loss = loss.item()
            count +=1
            print("{} loss {}".format(count, loss.item()))
            if loss.item()<0.1:
                break
        return

    def test(self, test_set):
        predicts = []
        curvatures = []
        true_vals = []

        for idx in range(len(test_set) - self.input_size - 1):

            preview = self.previewHelper.get_preview(idx, self.previewType)

            ks = preview['Curvature'][:self.input_size]
            augment_elem = [preview['GPS_Speed'][0] * self.rescale_velocity]
            if self.use_throttle:
                augment_elem = [preview['ECU_THROTTLE'][0] * self.rescale_throttle] + augment_elem
            if self.use_steer_spd:
                augment_elem = [preview['ECU_STEER_SPD'][0] * self.rescale_steer_spd] + augment_elem

            ks = np.concatenate((augment_elem, ks))
            predict = self.predictor(ks)

            curvatures.append(ks)
            predicts.append(predict.cpu().detach().numpy())
            true_vals.append(np.concatenate((preview['ECU_STEER_SPD'][:self.output_size],preview['GPS_Speed'][:self.output_size])))

            if idx % 100 == 0:
                print(ks, predict)

        return np.array(predicts), np.array(true_vals), curvatures, len(predicts)

    def save(self, filename):
        torch.save(self.predictor.state_dict(), '{}.pth'.format(filename))

    def load(self, filename):
        # torch.load_state_dict(self.predictor.state_dict(), '{}.pth'.format(filename))
        self.predictor.load_state_dict(torch.load("learned/{}.pth".format(filename)))
