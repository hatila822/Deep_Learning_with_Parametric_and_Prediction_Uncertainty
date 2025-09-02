# --- Inference with MC sampling + wavelet reconstruction (cleaned) ---

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader
import pywt

from MPNN_LSTM import MPNN_LSTM


# -----------------------------
# Wavelet reconstruction utils
# -----------------------------
def wavelet_reconstruct(prediction_2d, u_peak_mean, length_full=12001, resl_u=3, wavelet_fun="db6"):
    """
    Reconstruct full-length signals from multiresolution coefficients (approximation path).
    prediction_2d: (T, D) array of latent predictions per mode
    u_peak_mean:   (D,) scaling vector
    Returns: (length_full, D) reconstructed array
    """
    T, D = prediction_2d.shape
    out = np.zeros((length_full, D), dtype=np.float64)
    for md in range(D):
        out[:, md] = pywt.upcoef(
            part="a",
            coeffs=prediction_2d[:, md],
            wavelet=wavelet_fun,
            level=resl_u,
            take=length_full,
        )
    out *= u_peak_mean  # scale back to physical units
    return out


@torch.no_grad()
def mc_predict_full(model, a, b, c, d, e, f, num_samples, u_peak_mean, length_full=12001, resl_u=3, wavelet_fun="db6"):
    """
    Monte Carlo prediction with Concrete Dropout-enabled model.
    Returns:
        pred_std_full: (length_full, D)  std over MC samples after wavelet reconstruction
        pred_mean_full: (length_full, D) mean over MC samples after wavelet reconstruction
    """
    model.eval()  # ConcreteDropout still samples via torch.rand_like; eval is fine here

    # Forward passes
    preds, preds_log_var = [], []
    for _ in range(num_samples):
        y_hat, log_var_hat, _ = model(a, b, c, d, e, f)  # (1, T, D)
        preds.append(y_hat)
        preds_log_var.append(log_var_hat)

    preds = torch.stack(preds, dim=0)          # (S, 1, T, D)
    preds_log_var = torch.stack(preds_log_var)  # (S, 1, T, D)

    # Collapse batch=1
    preds = preds.squeeze(1)                   # (S, T, D)
    preds_log_var = preds_log_var.squeeze(1)   # (S, T, D)

    S, T, D = preds.shape
    device = preds.device

    # Epistemic via sample variance; aleatoric via mean(exp(log_var))
    mean_latent = preds.mean(dim=0)            # (T, D)
    var_ep_lat = preds.var(dim=0, unbiased=False)  # (T, D)
    var_model_lat = preds_log_var.exp().mean(dim=0)  # (T, D)

    std_total = (var_ep_lat + var_model_lat).sqrt()  # (T, D)

    # Reparameterize for full-field sampling (optional but keeps your logic)
    # mean/stdev broadcast to (S, T, D)
    mean_rep = mean_latent.unsqueeze(0).expand(S, -1, -1)
    std_rep = std_total.unsqueeze(0).expand(S, -1, -1)

    eps = torch.randn((S, 1, 1), device=device).expand(S, T, D)  # single epsilon per sample (as you had)
    sampled_latent = mean_rep + std_rep * eps                     # (S, T, D)

    # Move to CPU/NumPy for wavelet (pywt is NumPy-based)
    sampled_latent_np = sampled_latent.cpu().numpy().astype(np.float64)
    u_peak_mean_np = np.asarray(u_peak_mean, dtype=np.float64).squeeze()
    assert u_peak_mean_np.shape[0] == D, "U_Peak_mean length must match number of modes (D)."

    # Reconstruct each MC sample to full length
    pred_full_list = np.zeros((S, length_full, D), dtype=np.float64)
    for i in range(S):
        pred_full_list[i] = wavelet_reconstruct(
            sampled_latent_np[i], u_peak_mean_np, length_full=length_full, resl_u=resl_u, wavelet_fun=wavelet_fun
        )

    pred_std_full = pred_full_list.std(axis=0)
    pred_mean_full = pred_full_list.mean(axis=0)
    return pred_std_full, pred_mean_full


# -----------------------------
# Load tensors
# -----------------------------
inputs = torch.load('inputs.pt')        # (N, T, 38)
node_feature = torch.load('node_feature.pt')  # (N, N_nodes, 3)
outputs = torch.load('outputs.pt')      # (N, T, D)

R_r_T = torch.load('R_r_T.pt')          # (N, E, N_nodes)
R_r   = torch.load('R_r.pt')            # (N, N_nodes, E)
R_s   = torch.load('R_s.pt')            # (N, E, N_nodes)
R_c   = torch.load('R_c.pt')            # (N, N_nodes, N_nodes)

# Scaling vector
U_Peak_mean = scipy.io.loadmat('U_Peak_mean.mat')['U_Peak_mean'].squeeze()

# -----------------------------
# Inference ranges / output dirs
# -----------------------------
out_dir_std = "GNN_2000_Drop_STD"
out_dir_mean = "GNN_2000_Drop"
os.makedirs(out_dir_std, exist_ok=True)
os.makedirs(out_dir_mean, exist_ok=True)

num_sample = 2200
num_train_sample = 2000
batch_size = 5  # not used below, but kept for traceability

# Sanity checks
N = inputs.shape[0]
assert all(t.shape[0] >= num_sample for t in (inputs, node_feature, outputs, R_r_T, R_r, R_s, R_c)), \
    "One or more tensors have fewer than num_sample samples."
T, D = inputs.shape[1], outputs.shape[2]
N_nodes = node_feature.shape[1]
E = R_r.shape[2]

# -----------------------------
# Load model/checkpoint
# -----------------------------
model = MPNN_LSTM.load_from_checkpoint(
    checkpoint_path=r"C:\Users\samue\PycharmProjects\Interaction_LSTM\checkpoints\best\NLL_GNN_37.ckpt"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -----------------------------
# Per-sample inference loop
# -----------------------------
# We’ll predict for indices [2200, 2399] (200 samples), as in your original code.
start_idx = 2200
num_to_process = 200
num_mc = 50
length_full = 12001  # original full signal length
resl_u = 3
wavelet_fun = "db6"

for k in range(num_to_process):
    idx = start_idx + k
    print(f"Processing {k+1}/{num_to_process} (global idx {idx})")

    # Slice and add batch dimension without hard-coded shapes
    a = node_feature[idx].unsqueeze(0).to(device)        # (1, N_nodes, 3)
    b = inputs[idx].unsqueeze(0).to(device)              # (1, T, 38)
    c = R_r[idx].unsqueeze(0).to(device)                 # (1, N_nodes, E)
    d = R_r_T[idx].unsqueeze(0).to(device)               # (1, E, N_nodes)
    e = R_s[idx].unsqueeze(0).to(device)                 # (1, E, N_nodes)
    f = R_c[idx].unsqueeze(0).to(device)                 # (1, N_nodes, N_nodes)

    # MC prediction and wavelet reconstruction
    pred_std, pred_mean = mc_predict_full(
        model, a, b, c, d, e, f,
        num_samples=num_mc,
        u_peak_mean=U_Peak_mean,
        length_full=length_full,
        resl_u=resl_u,
        wavelet_fun=wavelet_fun,
    )

    # Save .mat files
    scipy.io.savemat(os.path.join(out_dir_std,  f"LSTM_Pred_STD_{k+1}.mat"),  {'pred_std':  pred_std})
    scipy.io.savemat(os.path.join(out_dir_mean, f"LSTM_Pred_{k+1}.mat"),      {'pred_mean': pred_mean})
