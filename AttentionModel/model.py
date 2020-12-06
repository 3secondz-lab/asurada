import pdb

import torch
from torch import nn
from torch.nn import functional as tf
from constants import device, hiddenDimension


class Encoder(nn.Module):
    """
    Encoder. (for curvature information)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, stride=2, padding=5, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, stride=1, padding=3, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=hiddenDimension, kernel_size=7, stride=1, padding=3, padding_mode='replicate')  # out_channels = encoder_dim
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
        self.window_width = 2  # 양 옆으로 2씩 보는데, 그러면 window 전체를 5, 실제 curvature에 mapping하면 10

    def forward(self, encoder_out, decoder_hidden, currentPosition):
        """
        Forward propagation.

        :param encoder_out: encoded curvatures, a tensor of dimension ()
        :param decoder_hidden: decoder output, a tensor of dimention ()
        :param currentPosition: 현재 위치 0에서부터 얼마나 이동했는지, 이 값은 그냥 supervision으로 활용하고자 함
        :return: attention weighted encoding, weights
        """
        B = encoder_out.size(0)  # batch_size
        S = encoder_out.size(1)  # source hidden state sequence length (for local attention)

        # # Aligned position (Local-p) =========================================
        aligned_position = S * self.sigmoid(self.vp(self.tanh(self.Wp(decoder_hidden))))

        # Alignment function (General) =========================================
        att = torch.bmm(self.Wa(encoder_out), decoder_hidden.unsqueeze(-1))
        alpha = self.softmax(att.squeeze(-1))

        # Locally focused alpha ================================================
        temp = torch.arange(0, S, dtype=torch.float32).unsqueeze(0).expand(B, -1).to(device)
        temp = temp - aligned_position  # to make it gaussian distribution
        temp = torch.exp(-(temp**2)/(self.window_width**2)/2)  # sigma: D/2
        alpha = alpha * temp

        awe = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return awe, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, encoder_dim, lstm_input_dim, decoder_dim, attention_dim, output_dim):
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
                vMean=None, vStd=None, aMean=None, aStd=None, inferenceMode=False, prepHistLen=None):
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
        alphas_target = torch.zeros(batch_size, decodeLength, numSubCurvatures).to(device)  # predicted position

        if inferenceMode:  # 이때는 항상 curSpeeds.dim()이 1이다. Not Used.
            histLen = histSpeeds.size(1) - prepHistLen

            for t in range(prepHistLen):
                speed = histSpeeds[0, t:t+histLen+1]
                accelX = histAccelXs[0, t:t+histLen+1]

                carState = torch.cat((speed.unsqueeze(0), accelX.unsqueeze(0)), 0)
                carState = carState.unsqueeze(0)
                csEncOut = self.relu(self.conv1(carState))
                csEncOut = self.relu(self.conv2(csEncOut))
                csEncOut = self.conv3(csEncOut)
                csEncOut = self.maxPool(csEncOut)  # (batch_size, 16, 1)

                if t == 0:
                    h, c = self.init_hidden_state(cEncOut, csEncOut)

                awe, alpha = self.attention(cEncOut, h)

                h, c = self.decode_step(torch.cat((csEncOut.squeeze(2), awe), 1), (h, c))
            histSpeeds = histSpeeds[:, -histLen:]
            histAccelXs = histAccelXs[:, -histLen:]

        currentPosition = torch.zeros([batch_size]).to(device)
        for position in currentPosition:
            alphas_target[:, 0, int(position.item())] = 1  # t=0일때 alphas_target
        # alphas_target = alphas_target.to(device)

        for t in range(decodeLength):
            if curSpeeds.dim() > 1:
                curSpeed = curSpeeds[:, t]  # 정답값으로 학습
                # curAccelX = curAccelXs[:, t]  # Not used
            else:
                if t == 0:
                    curSpeed = curSpeeds
                    # curAccelX = curAccelXs  # Not used

            speed = torch.cat((histSpeeds, curSpeed.unsqueeze(1)), 1)
            carState = speed

            if t == 0:
                if not inferenceMode:  # inferenceMode면, 이미 위에서 계산된 h가 있음.
                    h, c = self.init_hidden_state(cEncOut, carState)

            ## h_t -> a_t -> c_t -> ~h_t
            h, c = self.decode_step(carState, (h, c))

            awe, alpha = self.attention(cEncOut, h, currentPosition)

            preds = self.fc_output(torch.cat((awe, h), 1))

            curSpeed_t = (curSpeed*vStd+vMean)
            curSpeed_t1 = curSpeed_t + (preds*aStd + aMean).squeeze(-1)
            currentPosition = currentPosition + (curSpeed_t + 0.5*(abs(curSpeed_t1-curSpeed_t)))/36

            predictions[:, t, :] = preds  # [t=0] ^a_t --> ^v_(t+1) = v_t + ^a_t
            alphas[:, t, :] = alpha  # ^a_t를 예측할때 focusing 한 곡률 부분

            if t < (decodeLength-1):
                for b, position in enumerate(currentPosition):
                    try:
                        alphas_target[b, t+1, int(torch.round(position/2).item())] = 1  # /2 -> feature의 개수가 125개라서.
                    except:
                        alphas_target[b, t+1, numSubCurvatures-1] = 1

            if curSpeeds.dim() == 1:
                if batch_size == 1:
                    curSpeed = (((curSpeed*vStd+vMean) + (preds*aStd + aMean)) - vMean)/vStd
                    curSpeed = curSpeed.squeeze(-1)
                else:
                    curSpeed = ((curSpeed*vStd+vMean).unsqueeze(-1) + (preds*aStd+aMean) - vMean) / vStd
                    curSpeed = curSpeed.squeeze(-1)

            histSpeeds = speed[:, 1:]

        return predictions, alphas, alphas_target
