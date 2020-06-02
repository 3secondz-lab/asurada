import pdb
import torch

from network import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

class Model:
    def __init__(self, model):
        self.input_size = 100
        self.h1 = 100
        self.h2 = 64
        self.output_size = 30
        self.lr = 0.2

        self.use_throttle = False
        self.use_steer_spd = False

        self.rescale_velocity = 10 ** (-6)
        self.rescale_throttle = 10 ** (-6)
        self.rescale_steer_spd = 10 ** (-3)

        self.predictor = NetModel(self.input_size+self.use_throttle+self.use_steer_spd+1,
                                  self.h1, self.h2,
                                  self.output_size*(1+self.use_steer_spd), self.lr)

        self.predictor.load_state_dict(torch.load(model))

    def predict(self, curvatures, currentSpeed, currentThrottle, currentSteer):
        ks = curvatures[:self.input_size]
        augment_elem = [currentSpeed * self.rescale_velocity]
        if self.use_throttle:
            augment_elem = [currentThrottle * self.rescale_throttle] + augment_elem
        if self.use_steer_spd:
            augment_elem = [currentSteer * self.rescale_steer_spd] + augment_elem

        ks = np.concatenate((augment_elem, ks))
        predict = self.predictor(ks)

        return predict
