import pdb

import torch
from torch import nn
from torch.nn import functional as tf
from constants import device

# def init_hidden(x)

class Encoder(nn.Module):

    def __init__(self, input_size, enc_dim):
        super(Encoder, self).__init__()  # ?
        self.input_size = input_size
        self.enc_dim = enc_dim

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=enc_dim, num_layers=1)

    def forward(self, input_data):
        input_data = input_data.unsqueeze(2).permute(1, 0, 2)
        output, (hidden, cell) = self.lstm_layer(input_data)  # input: (seq_len, batch, input_size)
        # h, c는 값을 안주면 default initial: zero

        return output.permute(1, 0, 2)  # (batch, seq_len, enc_dim)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super(Attention, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.att_dim = att_dim

        self.att_linaer = nn.Linear(in_features=dec_dim+enc_dim, out_features=1)

    def forward(self, dec_h, enc_hs):

        enc_seq_len = enc_hs.size(1)
        dec_h = dec_h.unsqueeze(0).repeat(enc_seq_len, 1, 1).permute(1, 0, 2)

        x = torch.cat((dec_h, enc_hs), dim=2)
        x = self.att_linaer(x.view(-1, self.dec_dim + self.enc_dim))
        # pdb.set_trace()
        att_weight = tf.softmax(x.view(-1, enc_seq_len), dim=1)
        attended = torch.bmm(att_weight.unsqueeze(1), enc_hs).squeeze()

        return attended, att_weight

class Decoder(nn.Module):

    def __init__(self, enc_dim, dec_dim, att_dim, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim  # speed
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.att_dim = att_dim  # 사실상 여기서 att_dim은 enc_dim과 같음.

        self.attention = Attention(enc_dim, dec_dim, att_dim)
        self.decode_step = nn.LSTMCell(output_dim, dec_dim, bias=True)
        self.fc = nn.Linear(self.dec_dim + self.enc_dim + self.enc_dim, self.output_dim)

    # def forward(self, enc_hs, targetSpeeds, currentSpeed=None, ecodeLength=None):  # 이부분을 train일때랑 test일때 다 할 수 있도록 해야겠다.
    def forward(self, enc_hs_d, enc_hs_c, currentSpeed, decodeLength):

        # batch_size = targetSpeeds.size(0)
        # decodeLength = targetSpeeds.size(1) - 1

        batch_size = currentSpeed.size(0)  # (batch,)의 크기일 거 같음. input_size가 1이어서.

        enc_seq_len_d = enc_hs_d.size(1)
        enc_seq_len_c = enc_hs_c.size(1)

        preds = torch.zeros(batch_size, decodeLength).to(device)
        alphas_d = torch.zeros(batch_size, decodeLength, enc_seq_len_d).to(device)
        alphas_c = torch.zeros(batch_size, decodeLength, enc_seq_len_c).to(device)

        for t in range(decodeLength):
            # pdb.set_trace()
            if t == 0:
                # h, c = self.decode_step(targetSpeeds[:, 0].unsqueeze(1))  # input: (batch, input_size)
                h, c = self.decode_step(currentSpeed.unsqueeze(1), (enc_hs_d[:, -1, :], enc_hs_d[:, -1, :]))  # input: (batch, input_size)
            else:
                h, c = self.decode_step(pred, (h, c))

            # attention layer for driving context
            attended_d, alpha_d = self.attention(h, enc_hs_d)

            # attention layer for curvature context
            attended_c, alpha_c = self.attention(h, enc_hs_c)

            if batch_size == 1:
                pred = self.fc(torch.cat((h, attended_d.unsqueeze(0), attended_c.unsqueeze(0)), dim=1))
            else:
                pred = self.fc(torch.cat((h, attended_d, attended_c), dim=1))

            preds[:, t] = pred.squeeze()
            alphas_d[:, t, :] = alpha_d
            alphas_c[:, t, :] = alpha_c

        return preds, alphas_d, alphas_c
