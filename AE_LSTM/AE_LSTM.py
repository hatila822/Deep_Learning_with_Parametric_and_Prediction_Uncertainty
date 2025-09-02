import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import Adam


class ConcreteDropout(nn.Module):
    def __init__(self):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = 1e-6
        self.dropout_regularizer = 1e-5

        init_min = np.log(0.1) - np.log(1.0 - 0.1)
        init_max = np.log(0.1) - np.log(1.0 - 0.1)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)

        out, _ = layer(self._concrete_dropout(x, p))

        sum_of_square = sum((param ** 2).sum() for param in layer.parameters())
        weights_regularizer = self.weight_regularizer * sum_of_square / (1.0 - p)

        dropout_regularizer = p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
        input_dimensionality = x[0].numel()  # number of elements in first item of batch
        dropout_regularizer = dropout_regularizer * self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    @staticmethod
    def _concrete_dropout(x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)
        drop_prob = (
            torch.log(p + eps) - torch.log(1.0 - p + eps)
            + torch.log(unif_noise + eps) - torch.log(1.0 - unif_noise + eps)
        )
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob
        retain_prob = 1.0 - p

        x = x * random_tensor
        x = x / retain_prob
        return x


class AE_LSTM(pl.LightningModule):
    """
    4-layer LSTM with Concrete Dropout (on middle layers) and heteroscedastic loss.
    Outputs 40 features per time step.
    """

    def __init__(self):
        super().__init__()

        self.cd_2 = ConcreteDropout()
        self.cd_3 = ConcreteDropout()

        self.hidden_dim = 400
        self.num_layers = 4
        self.input_size = 200
        self.batch_first = True

        self.preW = nn.Linear(53, 128)
        self.preW2 = nn.Linear(128, self.input_size)

        self.lstm_1 = nn.LSTM(self.input_size,  self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_3 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_4 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

        self.preW4 = nn.Linear(self.hidden_dim, 40)
        self.log_W = nn.Linear(self.hidden_dim, 40)

    def forward(self, x: torch.Tensor):
        regularization = torch.empty(2, device=x.device)

        z = F.relu(self.preW(x))
        out = self.preW2(z)

        out, _ = self.lstm_1(out)
        out, regularization[0] = self.cd_2(out, self.lstm_2)
        out, regularization[1] = self.cd_3(out, self.lstm_3)
        out, _ = self.lstm_4(out)

        final = self.preW4(out)
        log_var = self.log_W(out)
        return final, log_var, regularization.sum()

    def configure_optimizers(self):
        decayRate = 0.8
        optimizer = Adam(self.parameters(), lr=0.25 * (10 ** (-3)))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[5, 10, 50, 75, 100, 200, 250, 300, 400, 500, 720, 750, 850,
                        900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 2750],
            gamma=decayRate
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def Define_Loss(self, pred, target, log_var):
        B, T, D = pred.shape
        gamma = B * T

        sigma2 = torch.exp(log_var).clamp(min=1e-6).reshape(gamma, D)
        log_var = log_var.reshape(gamma, D)
        pred = pred.reshape(gamma, D)
        target = target.reshape(gamma, D)

        diff_vector = (pred - target) ** 2
        loss_pre_weight = 0.5 * (diff_vector / sigma2 + log_var)

        weight = sigma2.sqrt().detach()
        loss = (loss_pre_weight * weight).sum() / (gamma * D)

        r_loss = diff_vector.sum() / (gamma * D)
        return loss, r_loss

    def training_step(self, batch, batch_idx):
        x_i, label_i = batch
        output_i, log_var_i, reg = self.forward(x_i)
        mse_loss, r_loss = self.Define_Loss(output_i, label_i, log_var_i)
        loss = mse_loss + reg
        self.log('train_loss', r_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_i, label_i = batch
        output_i, log_var_i, reg = self.forward(x_i)
        mse_loss, r_loss = self.Define_Loss(output_i, label_i, log_var_i)
        val_loss = mse_loss + reg
        self.log("val_Loss", r_loss, batch_size=x_i.size(0))
        return val_loss
