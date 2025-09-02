# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from MLP_LSTM import MLP_LSTM


seed_everything(3, workers=True)

torch.set_float32_matmul_precision('high')

inputs = torch.load('LSTM_inputs.pt')
outputs = torch.load('LSTM_outputs.pt')

num_sample = 2200
num_train_sample = 2000
batch_size = 5

train_dataset = TensorDataset(
    inputs[0:num_train_sample, :, :],
    outputs[0:num_train_sample, :, :]
)

val_dataset = TensorDataset(
    inputs[2000:2200, :, :],
    outputs[2000:2200, :, :]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

val_check_interval = len(train_loader)


model = MLP_LSTM()

callbacks = ModelCheckpoint(
    dirpath='checkpoints/best',
    save_top_k=1,
    mode='min',
    monitor="val_Loss"
)


lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=1200,
    val_check_interval=num_train_sample / batch_size,
    accelerator='gpu',
    log_every_n_steps=1,
    callbacks=[callbacks, lr_monitor]
)  # gradient_clip_val=0.2

trainer.fit(model, train_loader, val_loader)
callbacks.best_model_path

trainer.save_checkpoint("Concrete_LSTM_cat.ckpt")  # bigbatch
