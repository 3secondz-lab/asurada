import pdb
import torch

from model import Encoder, Decoder
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

class Model:
    def __init__(self, chpt_encC_path, chpt_encD_path, chpt_dec_path, chpt_stat_path):

        input_size=1
        hidden_size=64
        output_size=1

        self.encoder_c = Encoder(input_size=input_size, enc_dim=hidden_size)
        self.encoder_d = Encoder(input_size=input_size, enc_dim=hidden_size)
        self.decoder = Decoder(enc_dim=hidden_size, dec_dim=hidden_size, att_dim=hidden_size, output_dim=output_size)

        self.encoder_c.load_state_dict(torch.load(chpt_encC_path))
        self.encoder_d.load_state_dict(torch.load(chpt_encD_path))
        self.decoder.load_state_dict(torch.load(chpt_dec_path))

        self.encoder_c = self.encoder_c.to(device)
        self.encoder_d = self.encoder_d.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder_c.eval()
        self.encoder_d.eval()
        self.decoder.eval()

        with open(chpt_stat_path, 'rb') as f:
            chpt_stat = pickle.load(f)

        self.cMean = chpt_stat['cMean_tr']
        self.cStd = chpt_stat['cStd_tr']

        self.mean = chpt_stat['sMean_tr']
        self.std = chpt_stat['sStd_tr']

    def predict(self, curvatures, currentSpeed, histSpeeds):
        histSpeeds = torch.FloatTensor(histSpeeds).to(device)
        curvatures = torch.FloatTensor(curvatures).to(device)
        currentSpeed = torch.FloatTensor([currentSpeed]).to(device)

        decodeLength = 20

        enc_hiddens_c = self.encoder_c( (curvatures.unsqueeze(0)-self.cMean) / self.cStd)
        enc_hiddens_v = self.encoder_d( (histSpeeds.unsqueeze(0)-self.mean) / self.std)
        predictions, alphas_d, alphas_c = self.decoder(enc_hiddens_v, enc_hiddens_c, ((currentSpeed-self.mean) / self.std), decodeLength)

        return predictions*self.std + self.mean, alphas_d, alphas_c
