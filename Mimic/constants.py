import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hiddenDimension = 16
hiddenDimension = 32