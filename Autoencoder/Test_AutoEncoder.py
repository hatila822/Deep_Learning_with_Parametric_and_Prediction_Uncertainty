from Autoencoder import Autoencoder

#from Tran_LineGraphLSTM import Tran_LineGraphLSTM
# Imports From Torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Imports from Lightning
import lightning as L

# Other Imports
import numpy as np
#from tqdm import tqd
import scipy.io
from math import dist

# torch.set_float32_matmul_precision('medium' | 'high')
torch.set_float32_matmul_precision('high')

inputs = torch.load('LSTM_inputs.pt')
outputs = torch.load('LSTM_outputs.pt')

num_sample = 2200
num_train_sample = 2000
batch_size = 50

auto_input = torch.cat((inputs[:, :, 0:15], outputs[:, :, :]), dim=2)

train_dataset = TensorDataset(auto_input[0:num_train_sample, :, :],
                              outputs[0:num_train_sample, :, :])

val_dataset = TensorDataset(auto_input[num_train_sample:num_sample, :, :], outputs[num_train_sample:num_sample, :, :])

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = Autoencoder.load_from_checkpoint(
    checkpoint_path="C:/Users\samue\PycharmProjects\Interaction_LSTM\checkpoints/best\AutoEncoder_40.ckpt")

device = torch.device('cpu')

model.to(device)
a = auto_input[0:2400, :, :].to(device)

with torch.no_grad():
    full_space_prediction = model.Encoder(a)
    #full_space_prediction = torch.cat((full_space_prediction, a[:, :, 0:15]),dim=2)
    #full_space_prediction = model.Decoder(full_space_prediction)

# target = outputs[num_train_sample:num_sample, :, :]

##  For saving the result to plot in matlab
#target = outputs[0:2400, :, :]
file_path = 'AutoEncoder_Reduced_Space.mat' #'AutoEncoder_Reduced_Space.mat'
#file_path2 = 'AutoEncoder_target.mat'
scipy.io.savemat(file_path, {'AutoEncoder_Reduced_Space': full_space_prediction})
#scipy.io.savemat(file_path2, {'AutoEncoder_target': target})
print('Done')
