import pdb

import torch
from torch import nn
from torch.nn import functional as tf
from constants import device

import numpy as np

class Encoder(nn.Module):
    """
    Encoder. (for curvature information)
    """
    def __init__(self, encoder_dim):  # encoder_dim: 16
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, stride=2, padding=5, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, stride=1, padding=3, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=encoder_dim, kernel_size=7, stride=1, padding=3, padding_mode='replicate')
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, curvatures):
        """
        Forward propagation.

        :param curvatures: preview curvature vector, a tensor of dimensions (batch_size, 1, curvatureLength)
        :return: encoded curvatures, a tensor of dimensions (batch_size, encoder_dim, numSubCurvatures)
        """
        out = self.relu(self.conv1(curvatures))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out.permute(0, 2, 1)
        return out


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, attention_dim, decoder_dim):
        """
        :param encoder_dim:
        :param decoder_dim:
        :param attention_dim:
        """
        super(Attention, self).__init__()

        ''' Alignment Function (General)
            h_t^T * W_a * h_s '''
        self.Wa = nn.Linear(encoder_dim, decoder_dim)
        self.softmax = nn.Softmax(dim=1)

        ''' Local Attention
            p_t = S * sigmoid(v_p^T * tanh(W_p * h_t)) '''
        self.Wp = nn.Linear(decoder_dim, decoder_dim)
        self.vp = nn.Linear(decoder_dim, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.window_width = 1

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded curvatures, a tensor of dimension ()
        :param decoder_hidden: decoder output, a tensor of dimention ()
        :return: attention weighted encoding, weights
        """
        B = encoder_out.size(0)  # batch_size
        S = encoder_out.size(1)  # source hidden state sequence length (for local attention)

        # # Aligned position (Local-p)
        aligned_position = S * self.sigmoid(self.vp(self.tanh(self.Wp(decoder_hidden))))

        # Alignment function (General) =========================================
        att = torch.bmm(self.Wa(encoder_out), decoder_hidden.unsqueeze(-1))
        alpha = self.softmax(att.squeeze(-1))

        # # --> locally focused alpha
        temp = torch.arange(0, S, dtype=torch.float32).unsqueeze(0).expand(B, -1).to(device)
        temp = temp - aligned_position
        temp = torch.exp(-(temp**2)/(self.window_width**2)/2)  # sigma: D/2
        alpha = alpha * temp

        # # --> normalization
        # alpha_exp = torch.exp(alpha)
        # alpha_norm = (alpha_exp / alpha_exp.sum(1).unsqueeze(1).expand(-1, S))
        # # alpha_norm = (alpha/alpha.sum(1).unsqueeze(1).expand(-1, S))
        # # alpha = (alpha/alpha.sum(1).unsqueeze(1).expand(-1, S))
        # alpha_norm = alpha

        awe = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # awe = (encoder_out * alpha_norm.unsqueeze(2)).sum(dim=1)

        return awe, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, encoder_dim, lstm_input_dim, decoder_dim, attention_dim, output_dim, previewLength):
        """
        :param encoder_dim: feature size of encoded curvature vector
        """
        super(DecoderWithAttention, self).__init__()

        self.output_dim = output_dim

        self.init_h = nn.Linear(encoder_dim+lstm_input_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim+lstm_input_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell

        self.decode_step = nn.LSTMCell(lstm_input_dim, decoder_dim, bias=True)

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.fc_output = nn.Linear(attention_dim + decoder_dim, output_dim)

        self.previewLength = previewLength

    def init_hidden_state(self, cEncOut, csEncOut):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded curvature vector.

        :param encoder_out: encoded curvature vector, a tensor of dimension (batch_size, numSubCurvatures, encoder_dim)
        :return: hidden state, cell state
        """
        mean_cEncOut = cEncOut.mean(dim=1)

        h = self.init_h(torch.cat((mean_cEncOut, csEncOut), dim=1))
        c = self.init_c(torch.cat((mean_cEncOut, csEncOut), dim=1))

        return h, c

    def forward(self, cEncOut, curSpeeds, histSpeeds, curAccelXs, histAccelXs, decodeLength,
                vMean=None, vStd=None, aMean=None, aStd=None):
        """
        Forward propagation.

        :param cEncOut: encoded curvature vectors, a tensor of dimension (batch_size, 1, encoder_dim)
        :param curSpeeds:
        :param histSpeeds
        :param curAccelXs:
        :param histAccelXs
        :param decodeLength
        :return: scores
        """

        if curSpeeds.dim() == 1:
            assert vMean is not None, 'It needs vMean, vStd, aMean, aStd'

        batch_size = cEncOut.size(0)
        numSubCurvatures = cEncOut.size(1)

        predictions = torch.zeros(batch_size, decodeLength, self.output_dim).to(device)
        alphas = torch.zeros(batch_size, decodeLength, numSubCurvatures).to(device)

        for t in range(decodeLength):
            if curSpeeds.dim() > 1:
                curSpeed = curSpeeds[:, t]  # 정답값으로 학습
                # curAccelX = curAccelXs[:, t]
            else:
                if t == 0:
                    curSpeed = curSpeeds
                    # curAccelX = curAccelXs

            speed = torch.cat((histSpeeds, curSpeed.unsqueeze(1)), 1)
            carState = speed

            if t == 0:
                h, c = self.init_hidden_state(cEncOut, carState)

            ## h_t -> a_t -> c_t -> ~h_t
            h, c = self.decode_step(carState, (h, c))

            awe, alpha = self.attention(cEncOut, h)

            preds = self.fc_output(torch.cat((awe, h), 1))

            predictions[:, t, :] = preds  # [t=0] ^a_t --> ^v_(t+1) = v_t + ^a_t
            alphas[:, t, :] = alpha  # ^a_t를 예측할때 focusing 한 곡률 부분

            if curSpeeds.dim() == 1:
                if batch_size == 1:
                    curSpeed = (((curSpeed*vStd+vMean) + (preds*aStd + aMean)) - vMean)/vStd
                    curSpeed = curSpeed.squeeze(-1)
                else:
                    curSpeed = ((curSpeed*vStd+vMean).unsqueeze(-1) + (preds*aStd+aMean) - vMean) / vStd
                    curSpeed = curSpeed.squeeze(-1)

            histSpeeds = speed[:, 1:]

        return predictions, alphas