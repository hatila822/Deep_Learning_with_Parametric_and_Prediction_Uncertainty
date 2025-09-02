# Imports
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import pywt

from MLP_LSTM import MLP_LSTM


def wavelet_reconstruct(prediction, U_Peak_mean):
    N = 12001
    resl_u = 3
    wavelet_fun = 'db6'
    nmode = 798

    Pred_full = np.zeros((N, nmode))
    for md in range(nmode):
        # reconstruct the full‐length signal from the level‐resl_u approx. coeffs:
        Pred_full[:, md] = pywt.upcoef(
            part='a',          # 'a' = approximation path
            coeffs=prediction[:, md],
            wavelet=wavelet_fun,
            level=resl_u,
            take=N            # trim/pad to original length
        )
    Pred_full = Pred_full * U_Peak_mean
    return Pred_full


def mc_predict_full(model, x, num_samples, means_val):
    model.eval()  # keep dropout ON

    preds = []
    preds_var = []

    with torch.no_grad():
        for _ in range(num_samples):
            y_hat, log_var_hat, _reg = model(x)  # unpack
            preds.append(y_hat)
            preds_var.append(log_var_hat)

    preds = torch.stack(preds)       # [S, 1, 1509, 798]
    preds_var = torch.stack(preds_var)

    preds = torch.reshape(preds, (num_samples, 1509, 798))
    preds_var = torch.reshape(preds_var, (num_samples, 1509, 798))

    mean_laten = preds.mean(0)                     # [1509, 798]
    mean_laten = mean_laten.unsqueeze(0).expand(num_samples, -1, -1)

    var_ep_laten = preds.var(0)                    # epistemic
    var_model_laten = torch.exp(preds_var.mean(0)) # aleatoric (mean sigma^2)

    std_total = (var_ep_laten + var_model_laten).sqrt()
    std_total = std_total.unsqueeze(0).expand(num_samples, -1, -1)

    # sample noise on same device as x
    random_eps = torch.randn((num_samples, 1, 1), device=x.device).expand(num_samples, 1509, 798)
    sampled_laten = mean_laten + std_total * random_eps

    sampled_laten = sampled_laten.cpu().numpy()
    pred_full_list = np.zeros((num_samples, 12001, 798))

    for i in range(num_samples):
        pred_full_list[i] = wavelet_reconstruct(sampled_laten[i], means_val)

    pred_std = pred_full_list.std(0)   # epistemic + aleatoric combined (after recon)
    pred_mean = pred_full_list.mean(0)
    return pred_std, pred_mean


# Load data
inputs = torch.load('LSTM_inputs.pt')
outputs = torch.load('LSTM_outputs.pt')
U_Peak_mean = scipy.io.loadmat('U_Peak_mean.mat')['U_Peak_mean'].squeeze()

out_dir = "LSTM_2000_Drop_STD"
out_dir_2 = "LSTM_2000_Drop"
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

# Load model
model = MLP_LSTM.load_from_checkpoint(
    checkpoint_path=r"C:\Users\samue\PycharmProjects\Interaction_LSTM\checkpoints\best\NLL_LSTM_37.ckpt"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inference loop
for idx in range(200):
    print(idx)

    b = inputs[idx + 2200].to(device)         # [1509, 53]
    b = torch.reshape(b, (1, 1509, 53))       # [1, 1509, 53]

    pred_std, pred_mean = mc_predict_full(model, b, 50, U_Peak_mean)

    scipy.io.savemat(os.path.join(out_dir,  f"LSTM_Pred_STD_{idx + 1}.mat"),  {'pred_std':  pred_std})
    scipy.io.savemat(os.path.join(out_dir_2, f"LSTM_Pred_{idx + 1}.mat"),      {'pred_mean': pred_mean})

# # Optionally save target example:
# target = outputs[0:1].cpu().numpy().reshape(1509, 798)
# target = wavelet_reconstruct(target, U_Peak_mean)
# scipy.io.savemat('BAY_target.mat', {'BAY_target': target})
