import math

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
import scipy.io


class Autoencoder(L.LightningModule):

    def __init__(self):
        super(Autoencoder, self).__init__()

        final_dimension = 40  # The final dimension of the reduced space

        # Create Encoder (798 is the number of node in full space)

        self.Encoder = torch.nn.Sequential(
            torch.nn.Linear(813, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, final_dimension),
        )

       # self.inter = torch.nn.Sequential(
       #     torch.nn.Linear(final_dimension+10, 5),
       #     torch.nn.ReLU(),
       #     torch.nn.Linear(5, 5)
       # )

        # Create Decoder
        self.Decoder = torch.nn.Sequential(
            torch.nn.Linear(final_dimension+15, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 798),
        )

        self.learning_rate = 0.001

    def forward(self, x):
        batch_size, seq_len, node_num = x.size()
        gamma = batch_size * seq_len

        # Reshape the input
        x = x.view(gamma, node_num)

        z = self.Encoder(x)
        z = torch.cat((z,x[:,0:15]),dim=1)
        #z = self.inter(z)
        z = self.Decoder(z)

        z = z.view(batch_size,seq_len,798)

        return z

    def configure_optimizers(self):
        decayRate = 0.8
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=[5, 10, 50, 75, 100, 190, 195, 200, 250,
                                                                     300, 400, 450, 480, 550, 670, 800, 900],
                                                         gamma=decayRate)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        # return Adam(self.parameters(), lr=self.learning_rate)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def Define_Loss(self, pred, target):

        shape = pred.shape
        num_batch = shape[0]
        num_sequence = shape[1]
        gamma = num_batch * num_sequence

        pred = pred.reshape(gamma, 798)
        target = target.reshape(gamma, 798)

        loss = (pred - target) ** 2
        loss = torch.sum(loss)
        loss = loss / (gamma * 798)

        return loss

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)

        train_loss = self.Define_Loss(output_i, label_i)

        self.log("Train_Loss", train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        batch_size = 5

        input_i, label_i = batch
        output_i = self.forward(input_i)

        val_loss = self.Define_Loss(output_i, label_i)

        self.log("val_Loss", val_loss, batch_size=batch_size)

        return val_loss

    def test_step(self, batch, batch_idx):
        batch_size = 1

        input_i, label_i = batch
        output_i = self.forward(input_i)

        test_loss = self.Define_Loss(output_i, label_i)
        """
        file_path = 'CNNLSTMResult.mat'
        scipy.io.savemat(file_path, {'CNNLSTMResult': output_i})
        """
        torch.save(output_i, 'CNNLSTMResult.pt')
        self.log("test_Loss", test_loss, batch_size=batch_size)

        return test_loss
