import pdb
import torch

# from model import Encoder, DecoderWithAttention
# from model_G import Encoder, DecoderWithAttention
from model_G_LA import Encoder, DecoderWithAttention
# from model_G_LA_TS import Encoder, DecoderWithAttention
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, chpt_enc_path, chpt_dec_path, chpt_stat_path):
        historyLength = 10
        # historyLength = 0  # v31, 32

        hiddenDimension = 32
        encoder_dim = hiddenDimension
        lstm_input_dim = historyLength + 1
        decoder_dim = hiddenDimension
        attention_dim = hiddenDimension
        output_dim = 1

        self.decodeLength = 20

        self.encoder = Encoder(hiddenDimension)
        self.decoder = DecoderWithAttention(encoder_dim, lstm_input_dim, decoder_dim, attention_dim, output_dim)

        self.encoder.load_state_dict(torch.load(chpt_enc_path))
        self.decoder.load_state_dict(torch.load(chpt_dec_path))

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder.eval()
        self.decoder.eval()

        with open(chpt_stat_path, 'rb') as f:
            chpt_stat = pickle.load(f)

        self.cMean = chpt_stat['cMean_tr']
        self.cStd = chpt_stat['cStd_tr']

        self.vMean = chpt_stat['vMean_tr']
        self.vStd = chpt_stat['vStd_tr']

        self.aMean = chpt_stat['aMean_tr']
        self.aStd = chpt_stat['aStd_tr']

        self.lMean = chpt_stat['lMean_tr']
        self.lStd = chpt_stat['lStd_tr']

        self.mean = torch.Tensor([self.vMean, self.aMean]).to(device)
        self.std = torch.Tensor([self.vStd, self.aStd]).to(device)

    def predict(self, curvatures, currentSpeed, histSpeeds, currentAccelX, histAccelXs):
        curvatures = torch.FloatTensor(curvatures).to(device)

        currentSpeed = torch.FloatTensor([currentSpeed]).to(device)
        histSpeeds = torch.FloatTensor(histSpeeds).to(device)

        currentAccelX = torch.FloatTensor([currentAccelX]).to(device)
        histAccelXs = torch.FloatTensor(histAccelXs).to(device)

        curvatures = (curvatures - self.cMean) / self.cStd
        # currentSpeed = (currentSpeed - self.vMean) / self.vStd
        # histSpeeds = (histSpeeds - self.vMean) / self.vStd
        currentSpeed = (currentSpeed - self.lMean) / self.lStd
        histSpeeds = (histSpeeds - self.lMean) / self.lStd
        currentAccelX = (currentAccelX - self.aMean) / self.aStd
        histAccelXs = (histAccelXs - self.aMean) / self.aStd

        # pdb.set_trace()
        curvatures = self.encoder(curvatures.unsqueeze(dim=0).unsqueeze(dim=0))
        predictions, alphas = self.decoder(curvatures, currentSpeed, histSpeeds.unsqueeze(dim=0), currentAccelX, histAccelXs.unsqueeze(dim=0),
                                            self.decodeLength, self.vMean, self.vStd, self.aMean, self.aStd)
        # predictions, alphas, alphas_target = self.decoder(curvatures, currentSpeed, histSpeeds.unsqueeze(dim=0), currentAccelX, histAccelXs.unsqueeze(dim=0),
        #                             self.decodeLength, self.vMean, self.vStd, self.aMean, self.aStd)

        # pdb.set_trace()
        # return (predictions.squeeze()*self.aStd + self.aMean).cpu().detach().numpy(), alphas.squeeze().cpu().detach().numpy()
        return (predictions.squeeze()*self.lStd + self.lMean).cpu().detach().numpy(), alphas.squeeze().cpu().detach().numpy()