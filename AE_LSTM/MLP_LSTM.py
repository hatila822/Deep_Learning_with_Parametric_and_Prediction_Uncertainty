import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout wrapper that perturbs inputs to a given layer and
    returns (layer_output, regularization_term).
    """
    def __init__(self):
        super().__init__()
        self.weight_regularizer = 1e-6
        self.dropout_regularizer = 1e-5

        # init logit(p) around p0 = 0.1
        p0 = 0.1
        init_logit = torch.log(torch.tensor(p0)) - torch.log(torch.tensor(1.0 - p0))
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_logit.item(), init_logit.item()))

    def forward(self, x: torch.Tensor, layer: nn.Module):
        p = torch.sigmoid(self.p_logit)

        # forward with concrete-dropout-perturbed input
        out, _ = layer(self._concrete_dropout(x, p))

        # weight regularizer
        sum_sq = sum(torch.sum(param ** 2) for param in layer.parameters())
        weights_regularizer = self.weight_regularizer * sum_sq / (1.0 - p)

        # dropout entropy regularizer
        dropout_regularizer = p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
        input_dimensionality = x[0].numel()  # elements in a single batch item
        dropout_regularizer = dropout_regularizer * self.dropout_regularizer * input_dimensionality

        return out, (weights_regularizer + dropout_regularizer)

    @staticmethod
    def _concrete_dropout(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 0.1

        u = torch.rand_like(x)
        logit = (
            torch.log(p + eps) - torch.log(1.0 - p + eps)
            + torch.log(u + eps) - torch.log(1.0 - u + eps)
        )
        drop_prob = torch.sigmoid(logit / temp)
        retain = 1.0 - drop_prob
        retain_prob = 1.0 - p

        x = x * retain
        x = x / retain_prob
        return x


class MLP_LSTM(pl.LightningModule):
    """
    4-layer LSTM with Concrete Dropout on intermediate layers and
    heteroscedastic loss (mean + log-variance heads).
    """
    def __init__(self):
        super().__init__()

        # ---- model dims (collect magic numbers) ----
        self.input_raw_dim = 53
        self.pre_hidden = 128
        self.input_size = 200
        self.hidden_dim = 400
        self.num_layers = 4
        self.n_outputs = 798  # output features per time step

        # ---- concrete dropout wrappers ----
        self.cd_2 = ConcreteDropout()
        self.cd_3 = ConcreteDropout()

        # ---- input projection ----
        self.preW = nn.Linear(self.input_raw_dim, self.pre_hidden)
        self.preW2 = nn.Linear(self.pre_hidden, self.input_size)

        # ---- LSTM stack ----
        self.lstm_1 = nn.LSTM(self.input_size,  self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_3 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_4 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

        # ---- output heads ----
        self.preW4 = nn.Linear(self.hidden_dim, self.n_outputs)  # mean
        self.log_W = nn.Linear(self.hidden_dim, self.n_outputs)  # log-variance

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, input_raw_dim)
        returns:
          final:   (B, T, n_outputs)
          log_var: (B, T, n_outputs)
          reg:     () scalar regularization
        """
        # project inputs
        z = F.relu(self.preW(x))
        out = self.preW2(z)

        # LSTM stack with Concrete Dropout on 2 & 3
        out, _ = self.lstm_1(out)

        reg_terms = torch.empty(2, device=out.device)
        out, reg_terms[0] = self.cd_2(out, self.lstm_2)
        out, reg_terms[1] = self.cd_3(out, self.lstm_3)

        out, _ = self.lstm_4(out)

        final = self.preW4(out)
        log_var = self.log_W(out)
        return final, log_var, reg_terms.sum()

    # ------------------ optim/sched ------------------
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2.5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[5, 10, 50, 75, 100, 200, 250, 300, 400, 500,
                        720, 750, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 2750],
            gamma=0.8,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    # ------------------ loss ------------------
    def Define_Loss(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor):
        """
        Heteroscedastic objective:
          0.5 * [ (pred - target)^2 / sigma^2 + log(sigma^2) ] * stop_grad(sigma)
        averaged over batch * time * outputs
        """
        B, T, D = pred.shape
        gamma = B * T

        sigma2 = log_var.exp().clamp(min=1e-6)   # (B, T, D)
        diff_sq = (pred - target) ** 2           # (B, T, D)

        base = 0.5 * (diff_sq / sigma2 + log_var)
        weight = sigma2.sqrt().detach()

        loss = (base * weight).sum() / (gamma * D)
        r_loss = diff_sq.sum() / (gamma * D)     # plain MSE for logging
        return loss, r_loss

    # ------------------ steps ------------------
    def training_step(self, batch, batch_idx):
        x_i, y_i = batch
        pred, log_var, reg = self.forward(x_i)
        hetero_loss, mse_log = self.Define_Loss(pred, y_i, log_var)
        loss = hetero_loss + reg
        self.log("train_loss", mse_log)
        return loss

    def validation_step(self, batch, batch_idx):
        x_i, y_i = batch
        pred, log_var, reg = self.forward(x_i)
        hetero_loss, mse_log = self.Define_Loss(pred, y_i, log_var)
        val_loss = hetero_loss + reg
        self.log("val_Loss", mse_log, batch_size=x_i.size(0))
        return val_loss
