from MPNN_LSTM import MPNN_LSTM

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# -----------------------------
# Reproducibility / precision
# -----------------------------
seed_everything(3, workers=True)
torch.set_float32_matmul_precision('high')  # keep your original choice

# -----------------------------
# Data loading
# -----------------------------
inputs = torch.load('inputs.pt')  # (N, T, 38)
node_feature = torch.load('node_feature.pt')  # (N, N_nodes, 3)
outputs = torch.load('outputs.pt')  # (N, T, 798)

R_r_T = torch.load('R_r_T.pt')  # (N, E, N_nodes)
R_r = torch.load('R_r.pt')  # (N, N_nodes, E)
R_s = torch.load('R_s.pt')  # (N, E, N_nodes)
R_c = torch.load('R_c.pt')  # (N, N_nodes, N_nodes)

# -----------------------------
# Split config
# -----------------------------
num_sample = 2200
num_train_sample = 2000
batch_size = 5

assert all(t.shape[0] >= num_sample for t in (inputs, node_feature, outputs, R_r_T, R_r, R_s, R_c)), \
    "One or more tensors have fewer than num_sample samples."

train_dataset = TensorDataset(
    node_feature[:num_train_sample],
    inputs[:num_train_sample],
    R_r[:num_train_sample],
    R_r_T[:num_train_sample],
    R_s[:num_train_sample],
    R_c[:num_train_sample],
    outputs[:num_train_sample],
)

val_dataset = TensorDataset(
    node_feature[num_train_sample:num_sample],
    inputs[num_train_sample:num_sample],
    R_r[num_train_sample:num_sample],
    R_r_T[num_train_sample:num_sample],
    R_s[num_train_sample:num_sample],
    R_c[num_train_sample:num_sample],
    outputs[num_train_sample:num_sample],
)

# You can tune num_workers/pin_memory for your machine
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False, drop_last=False)

# Validate once per epoch at the end (i.e., every len(train_loader) steps)
val_check_interval = len(train_loader)

# -----------------------------
# Model + callbacks
# -----------------------------
model = MPNN_LSTM()

ckpt_cb = ModelCheckpoint(
    dirpath='checkpoints/bestMPNN',
    save_top_k=1,
    mode='min',
    monitor='val_Loss'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# -----------------------------
# Trainer
# -----------------------------
trainer = pl.Trainer(
    max_epochs=1200,
    accelerator='gpu',  # uses GPU if available
    log_every_n_steps=1,
    val_check_interval=val_check_interval,  # validate at end of each epoch
    callbacks=[ckpt_cb, lr_monitor],
    enable_progress_bar=True,
)

# -----------------------------
# Train + save
# -----------------------------
trainer.fit(model, train_loader, val_loader)

#print("Best checkpoint:", ckpt_cb.best_model_path)

# Optional: save a final checkpoint (separate from "best")
trainer.save_checkpoint("Best_MPNN_LSTM.ckpt")
