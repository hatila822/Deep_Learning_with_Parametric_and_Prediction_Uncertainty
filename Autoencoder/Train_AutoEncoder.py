from Autoencoder import Autoencoder

# Imports From Torch
import torch

from torch.utils.data import TensorDataset, DataLoader

# Imports from Lightning
import lightning as L
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor

# device = torch.device('cpu')

seed_everything(3, workers=True)
# torch.set_float32_matmul_precision('medium' | 'high')
torch.set_float32_matmul_precision('high')

inputs = torch.load('LSTM_inputs.pt')
outputs = torch.load('LSTM_outputs.pt')

num_sample = 2200
num_train_sample = 2000
batch_size = 5

auto_input = torch.cat((inputs[:, :, 0:15], outputs[:, :, :]), dim=2)

train_dataset = TensorDataset(auto_input[0:num_train_sample, :, :],
                              outputs[0:num_train_sample, :, :])

val_dataset = TensorDataset(auto_input[num_train_sample:num_sample, :, :], outputs[num_train_sample:num_sample, :, :])

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

val_check_interval = len(train_loader)

model = Autoencoder()#.load_from_checkpoint(checkpoint_path="C:/Users\samue\PycharmProjects\Interaction_LSTM\Autoencoder.ckpt")

callbacks = ModelCheckpoint(dirpath='checkpoints/best', save_top_k=1, mode='min', monitor="val_Loss")
# callbacks_tr = ModelCheckpoint(dirpath='checkpoints/best_tr', save_top_k=1, mode='min', monitor="Train_Loss")


lr_monitor = LearningRateMonitor(logging_interval='step')

# early_stop_callback = EarlyStopping(monitor="loss/validate", stopping_threshold=1000000000.00, patience=3,
# verbose=False, mode="min")

"""
num_train_sample = 798
#num_train_sample / batch_size
"""

trainer = L.Trainer(max_epochs=5000, val_check_interval=num_train_sample / batch_size, accelerator='gpu',
                    log_every_n_steps=1,
                    callbacks=[callbacks, lr_monitor])  #gradient_clip_val=0.2

# trainer = L.Trainer(max_epochs=45,val_check_interval=val_check_interval, accelerator='cpu',log_every_n_steps=30,
# callbacks=[early_stop_callback])

trainer.fit(model, train_loader,
            val_loader)#, ckpt_path="C:/Users\samue\PycharmProjects\Interaction_LSTM/160_Trail_Autoencoder.ckpt")
callbacks.best_model_path

trainer.save_checkpoint("AA_Trail_Autoencoder.ckpt")  # bigbatch
