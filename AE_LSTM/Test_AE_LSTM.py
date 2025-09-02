# Imports From Torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from AE_LSTM import AE_LSTM
from Autoencoder import Autoencoder

# Imports from Lightning
import lightning as L

# Other Imports
import numpy as np
# from tqdm import tqd
import scipy.io
from math import dist
import pywt
import os
from joblib import Parallel, delayed



def wavelet_reconstruct(prediction, U_Peak_mean):
    N = 12001
    resl_u = 3
    wavelet_fun = 'db6'
    nmode = 798

    Pred_full = np.zeros((N, nmode))

    for md in range(nmode):
        # reconstruct the full‐length signal from the level‐resl_u approx. coeffs:
        Pred_full[:, md] = pywt.upcoef(
            part='a',  # 'a' = approximation path
            coeffs=prediction[:, md],
            wavelet=wavelet_fun,
            level=resl_u,
            take=N  # trim/pad to original length
        )
    Pred_full = Pred_full * U_Peak_mean
    return Pred_full


def mc_predict_full(model, x, num_samples, means_val, model_auto, mean_reduce, std_reduce):
    model.eval()   # keep dropout ON
    model_auto.eval()

    preds = []
    preds_var = []

    with torch.no_grad():
        for _ in range(num_samples):
            y_hat, log_var_hat, _reg = model(x)
            preds.append(y_hat)
            preds_var.append(log_var_hat)

    preds = torch.stack(preds)      # [S, 1, 1509, 40]
    preds_var = torch.stack(preds_var)

    preds = torch.reshape(preds, (num_samples, 1509, 40))
    preds_var = torch.reshape(preds_var, (num_samples, 1509, 40))

    mean_laten = preds.mean(0)                      # [1509, 40]
    mean_laten = mean_laten.unsqueeze(0).expand(num_samples, -1, -1)

    var_ep_laten = preds.var(0)
    var_model_laten = torch.exp(preds_var.mean(0))

    std_total = (var_ep_laten + var_model_laten) ** 0.5
    std_total = std_total.unsqueeze(0).expand(num_samples, -1, -1)

    # sample noise on same device as x
    random_eps = torch.randn((num_samples, 1, 1), device=x.device).expand(num_samples, 1509, 40)
    sampled_laten = mean_laten + std_total * random_eps

    # de-normalize reduced space
    sampled_laten = sampled_laten * std_reduce + mean_reduce

    # concatenate material uncertainty features (first 15 dims of input)
    mat_uncertainty = torch.tile(x[:, :, 0:15], (num_samples, 1, 1))
    preds = torch.cat((sampled_laten, mat_uncertainty), dim=2)  # [S, 1509, 55]

    preds = torch.reshape(preds, (num_samples * 1509, 55))
    preds = model_auto.Decoder(preds)
    preds = torch.reshape(preds, (num_samples, 1509, 798))

    preds = preds.cpu().detach().numpy()

    pred_full_list = np.zeros((num_samples, 12001, 798))
    for i in range(num_samples):
        pred_full_list[i, :, :] = wavelet_reconstruct(preds[i, :, :], means_val)

    pred_std = pred_full_list.std(0)
    pred_mean = pred_full_list.mean(0)
    return pred_std, pred_mean


inputs = torch.load('LSTM_inputs.pt')
outputs = torch.load('Autoencoder_outputs.pt')

mean_reduce = torch.load('auto_reduced_space_mean.pt')
std_reduce = torch.load('auto_reduced_space_std.pt')

# ensure tensors
if not isinstance(mean_reduce, torch.Tensor):
    mean_reduce = torch.from_numpy(mean_reduce)
if not isinstance(std_reduce, torch.Tensor):
    std_reduce = torch.from_numpy(std_reduce)

U_Peak_mean = scipy.io.loadmat('U_Peak_mean.mat')
U_Peak_mean = U_Peak_mean['U_Peak_mean'].squeeze()

out_dir = "Reduce_LSTM_2000_Drop_STD"
out_dir_2 = "Reduce_LSTM_2000_Drop"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_2, exist_ok=True)

num_sample = 2200
num_train_sample = 2000
batch_size = 5

train_dataset = TensorDataset(inputs[0:num_train_sample, :, :],
                              outputs[0:num_train_sample, :, :])

val_dataset = TensorDataset(inputs[num_train_sample:num_sample, :, :],
                            outputs[num_train_sample:num_sample, :, :])

train_loader = DataLoader(train_dataset, batch_size=5)
val_loader = DataLoader(val_dataset, batch_size=5)

model_LSTM = AE_LSTM.load_from_checkpoint(
    checkpoint_path=r"C:\Users\samue\PycharmProjects\Interaction_LSTM\checkpoints\best\NLL_Reduce_37.ckpt"
)

model_decoder = Autoencoder.load_from_checkpoint(
    checkpoint_path=r"C:\Users\samue\PycharmProjects\Interaction_LSTM\checkpoints\best\AutoEncoder_40.ckpt"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_LSTM.to(device)
model_decoder.to(device)
mean_reduce = mean_reduce.to(device)
std_reduce = std_reduce.to(device)

for idx in range(200):
    print(idx)

    b = inputs[(idx + 2200), :, :].to(device)
    b = torch.reshape(b, (1, 1509, 53))

    pred_std, pred_mean = mc_predict_full(
        model_LSTM, b, 50, U_Peak_mean, model_decoder, mean_reduce, std_reduce
    )

    fname = os.path.join(out_dir, f"LSTM_Pred_STD_{idx + 1}.mat")
    fname_2 = os.path.join(out_dir_2, f"LSTM_Pred_{idx + 1}.mat")

    scipy.io.savemat(fname,  {'pred_std':  pred_std})
    scipy.io.savemat(fname_2, {'pred_mean': pred_mean})

# target saving (optional)
# target = outputs[0:1, :, :].cpu().numpy().reshape(1509, 798)
# target = wavelet_reconstruct(target, U_Peak_mean)
# scipy.io.savemat('BAY_target.mat', {'BAY_target': target})
