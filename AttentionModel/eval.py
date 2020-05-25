import pdb

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from UtilsAtt import *
import argparse
from random import random
import matplotlib.pyplot as plt
from PIL import Image

# Parameters
data_folder = './previewData'  # folder with data files saved by create_input_files.py
data_name = 'std001_10_previewTime_0.5_sWindow'  # base name shared by data files
checkpoint = './BEST_checkpoint_{}.pth.tar'.format(data_name)  # model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
trMean = checkpoint['trMean']
trStd = checkpoint['trStd']

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def evaluate(vis=False):
    """
    Evaluation

    :return: mape, rmse
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        SpeedDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]),
        mean=trMean, std=trStd),
        batch_size=32, shuffle=True, num_workers=0, pin_memory=True)  # 1-> 0 (since window!)

    mapes = []
    rmses = []

    mean = loader.dataset.mean  # == train_loader.dataset.mean
    std = loader.dataset.std  # == train_loader.dataset.std

    print('\n===== EVALUATING... =====\n')
    # For each image
    for i, (imgs, target_speeds) in enumerate(tqdm(loader)):

        # Move to GPU device, if available
        imgs = imgs.to(device)  # (1, 3, 224, 224)
        target_speeds = target_speeds.to(device)

        # Encode
        encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
        batch_size = encoder_out.size(0)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        current_speed = target_speeds[:, 0].reshape(-1, 1)
        current_speed = (current_speed - mean) / std
        current_speed = current_speed.to(device)

        decode_length = target_speeds.size(1) - 1

        predictions = torch.zeros(batch_size, decode_length).to(device)
        alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)

        h, c = decoder.init_hidden_state(encoder_out)

        for t in range(decode_length):

            attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            if t == 0:
                h, c = decoder.decode_step(torch.cat([current_speed, attention_weighted_encoding], dim=1), (h, c))
            else:
                h, c = decoder.decode_step(torch.cat([preds, attention_weighted_encoding], dim=1), (h, c))

            preds = decoder.fc(h)

            predictions[:, t] = preds.squeeze(dim=1)
            alphas[:, t, :] = alpha

        targets = target_speeds[:, 1:]
        mape, rmse = accuracy(predictions*std+mean, targets*std+mean)

        mapes.append(mape)
        rmses.append(rmse)

        if vis:  # random visualization
            if random() > 0.8:
                imgIdx = 0
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                ax1.imshow(imgs[imgIdx].permute(1, 2, 0).cpu())

                pred_vis = predictions[imgIdx]
                target_vis = target_speeds[imgIdx]

                mape, rmse = accuracy(pred_vis*std+mean, target_vis[1:]*std+mean)

                pred_vis = torch.cat([target_vis[0].unsqueeze(dim=0), pred_vis])
                ax2.plot((pred_vis*std+mean).cpu().detach().numpy(), 'r.-', label='Predict')
                ax2.plot((target_vis*std+mean).cpu().detach().numpy(), 'b.-', label='True')
                ax2.set_title('MAPE:{:.4f}, RMSE:{:.4f}'.format(mape, rmse))

                plt.legend()
                plt.show()

    avgMAPE = sum(mapes)/len(mapes)
    avgRMSE = sum(rmses)/len(rmses)

    return avgMAPE, avgRMSE


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--vis', '-v')

    args = parser.parse_args()

    mape, rmse = evaluate(args.vis)

    print('Test Performance')
    print('\nMAPE: {:.4}'.format(mape))
    print('\nRMSE: {:.4}'.format(rmse))
