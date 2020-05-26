
import pdb

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

class dataNormalization:
    def __init__(self, data):
        self.mu, self.std = norm.fit(data)
        self.data = (data-self.mu)/self.std

def files2df(DATA_PATH, files):
    df_list = []
    for file in files:
        filepath = os.path.join(DATA_PATH, file)
        assert os.path.exists(filepath), 'No filename {}'.format(file)
        df_list.append(pd.read_csv(filepath))
    return pd.concat(df_list)
