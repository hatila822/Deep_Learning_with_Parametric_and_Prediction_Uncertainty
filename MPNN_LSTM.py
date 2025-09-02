import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as L


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout wrapper for an nn.Module (e.g., LSTM).
    Applies concrete dropout to the inputs and returns (layer_out, reg_term).
    """

    def __init__(self):
        super().__init__()
        self.weight_regularizer = 1e-6
        self.dropout_regularizer = 1e-5

        # Initialize logit(p) around a desired starting p (here 0.1)
        p0 = 0.1
        init_min = torch.log(torch.tensor(p0)) - torch.log(torch.tensor(1.0 - p0))
        init_max = init_min
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min.item(), init_max.item()))

    def forward(self, x: torch.Tensor, layer: nn.Module):
        p = torch.sigmoid(self.p_logit)

        # Forward through wrapped layer with concrete-dropout-perturbed input
        out, _ = layer(self._concrete_dropout(x, p))

        # L2 weight regularizer scaled by (1 - p)
        sum_of_square = sum(torch.sum(param ** 2) for param in layer.parameters())
        weights_regularizer = self.weight_regularizer * sum_of_square / (1.0 - p)

        # Dropout entropy regularizer scaled by input dimensionality
        dropout_regularizer = p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
        input_dimensionality = x[0].numel()  # elements in first item of batch (T * D for sequences)
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    @staticmethod
    def _concrete_dropout(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 0.1

        # Sample relaxed Bernoulli via Concrete distribution
        unif = torch.rand_like(x)
        logit = (
            torch.log(p + eps) - torch.log(1.0 - p + eps)
            + torch.log(unif + eps) - torch.log(1.0 - unif + eps)
        )
        drop_prob = torch.sigmoid(logit / temp)
        retain = 1.0 - drop_prob
        retain_prob = 1.0 - p

        x = x * retain
        x = x / retain_prob
        return x


class MPNN_LSTM(L.LightningModule):
    """
    MPNN + LSTM model with Concrete Dropout and heteroscedastic NLL loss.
    """

    def __init__(self):
        super().__init__()

        # ----- Core dimensions / constants -----
        self.hid_dim = 256
        self.hidden_dim = 400
        self.seq_feature_dim = 38      # last-dim of F_i (kept explicit for clarity)
        self.out_feature_dim = 200     # output of preW, input to first LSTM
        self.n_outputs = 798           # output dimension per time step

        # ----- MPNN encoder -----
        self.encoder = nn.Linear(3, self.hid_dim)  # encode node features (3 -> hid_dim)

        self.fc1m = nn.Linear(2 * self.hid_dim, self.hid_dim)
        self.fc2m = nn.Linear(self.hid_dim, self.hid_dim)

        self.fc1u = nn.Linear(2 * self.hid_dim, self.hid_dim)
        self.fc2u = nn.Linear(self.hid_dim, self.hid_dim)

        # ----- Sequence pre-projection before LSTMs -----
        self.preW = nn.Linear(self.hid_dim + self.seq_feature_dim, self.out_feature_dim)

        # ----- LSTM stack -----
        self.lstm_1 = nn.LSTM(input_size=self.out_feature_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=self.hidden_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1, batch_first=True)
        self.lstm_4 = nn.LSTM(input_size=self.hidden_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=1, batch_first=True)

        # ----- Concrete Dropout wrappers for middle LSTMs -----
        self.cd_2 = ConcreteDropout()
        self.cd_3 = ConcreteDropout()

        # ----- Output heads (mean and log-variance) -----
        self.fc = nn.Linear(self.hidden_dim, self.n_outputs)
        self.log_W = nn.Linear(self.hidden_dim, self.n_outputs)

    # ---------------------------
    # Graph message passing block
    # ---------------------------
    def update_node(
        self,
        node_feat: torch.Tensor,  # (B, N, H)
        F_i: torch.Tensor,        # (B, T, seq_feature_dim)  [not used inside; kept for signature parity]
        R_r: torch.Tensor,        # (B, N, E)
        R_r_T: torch.Tensor,      # (B, E, N)
        R_s: torch.Tensor,        # (B, E, N)
        R_c: torch.Tensor,        # (B, N, N)
        batch_size: int,
        node_num: int,
        edge_num: int,
    ) -> torch.Tensor:
        # Step 1: edge messages (receiver/sender)
        mg_r = torch.matmul(R_r_T, node_feat)  # (B, E, H)
        mg_s = torch.matmul(R_s,   node_feat)  # (B, E, H)
        mg = torch.cat((mg_r, mg_s), dim=2)    # (B, E, 2H)

        # Step 2: edge update
        new_E = self.fc2m(F.relu(self.fc1m(mg)))  # (B, E, H)

        # Step 3: aggregate to nodes
        E_bar = torch.matmul(R_r, new_E)                 # (B, N, H)
        m_i = torch.matmul(E_bar.transpose(1, 2), R_c)   # (B, H, N)
        m_i = m_i.transpose(1, 2)                        # (B, N, H)

        # Step 4: node update (residual outside)
        v_i = self.fc2u(F.relu(self.fc1u(torch.cat((node_feat, m_i), dim=2))))  # (B, N, H)
        return v_i

    def forward(
        self,
        O: torch.Tensor,       # (B, N, 3) node features
        F_i: torch.Tensor,     # (B, T, seq_feature_dim) sequence features
        R_r: torch.Tensor,     # (B, N, E)
        R_r_T: torch.Tensor,   # (B, E, N)
        R_s: torch.Tensor,     # (B, E, N)
        R_c: torch.Tensor,     # (B, N, N)
    ):
        # Shapes
        batch_size, node_num, num_feature = O.size()
        _, _, edge_num = R_r.size()
        _, seq_len, _ = F_i.size()

        # Encode node features
        node_feat_enc = self.encoder(O.reshape(batch_size, node_num, num_feature))  # (B, N, H)

        # 1 MPNN layer + 2 residual refinements
        z = self.update_node(node_feat_enc, F_i, R_r, R_r_T, R_s, R_c,
                             batch_size, node_num, edge_num)
        for _ in range(2):
            z = z + self.update_node(z, F_i, R_r, R_r_T, R_s, R_c,
                                     batch_size, node_num, edge_num)

        # Global mean pooling over nodes
        z = z.mean(dim=1)  # (B, H)

        # Tile pooled graph embedding across time, then concat with sequence features
        z = z.unsqueeze(1).expand(batch_size, seq_len, self.hid_dim)  # (B, T, H)
        z = torch.cat((z, F_i), dim=2)                                # (B, T, H + seq_feature_dim)

        # Project before LSTMs
        out = self.preW(z)  # (B, T, out_feature_dim)

        # LSTM stack with Concrete Dropout regularization on middle layers
        reg_terms = torch.empty(2, device=self.device)

        out, _ = self.lstm_1(out)
        out, reg_terms[0] = self.cd_2(out, self.lstm_2)
        out, reg_terms[1] = self.cd_3(out, self.lstm_3)
        out, _ = self.lstm_4(out)

        # Output heads (per time step)
        final = self.fc(out)       # mean
        log_var = self.log_W(out)  # log variance

        return final, log_var, reg_terms.sum()

    # ---------------------------
    # Optimizer / Scheduler
    # ---------------------------
    def configure_optimizers(self):
        decay_rate = 0.8
        optimizer = Adam(self.parameters(), lr=2.5e-4)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[
                5, 10, 50, 75, 100, 200, 250, 300, 400, 500,
                720, 750, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 2750
            ],
            gamma=decay_rate,
        )

        # Step the scheduler every training step (matches your original intent)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):

        scheduler.step(epoch=self.current_epoch)

    # ---------------------------
    # Loss
    # ---------------------------
    def Define_Loss(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor):
        """
        Heteroscedastic NLL-like objective with a stabilization term:
        loss = 0.5 * ((pred - target)^2 / sigma^2 + log(sigma^2)) * sqrt(sigma^2)
        (Averaged over batch * time * outputs)
        """
        B, T, _ = pred.shape
        gamma = B * T

        sigma2 = torch.exp(log_var).clamp(min=1e-6)          # (B, T, D)
        sigma2 = sigma2.reshape(gamma, self.n_outputs)
        log_var = log_var.reshape(gamma, self.n_outputs)
        pred = pred.reshape(gamma, self.n_outputs)
        target = target.reshape(gamma, self.n_outputs)

        diff_sq = (pred - target) ** 2
        base = (diff_sq / sigma2 + log_var) * 0.5

        weight = sigma2.sqrt().detach()                      # stop-gradient on weighting
        loss = (base * weight).sum() / (gamma * self.n_outputs)

        # Raw MSE for logging
        r_loss = diff_sq.sum() / (gamma * self.n_outputs)
        return loss, r_loss

    # ---------------------------
    # Steps
    # ---------------------------
    def training_step(self, batch, batch_idx):
        O_i, F_i, R_r_i, R_r_T_i, R_s_i, R_c_i, label_i = batch
        output_i, log_var_i, reg = self.forward(O_i, F_i, R_r_i, R_r_T_i, R_s_i, R_c_i)

        mse_loss, r_loss = self.Define_Loss(output_i, label_i, log_var_i)
        loss = mse_loss + reg

        self.log("train_loss", r_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        O_i, F_i, R_r_i, R_r_T_i, R_s_i, R_c_i, label_i = batch
        output_i, log_var_i, reg = self.forward(O_i, F_i, R_r_i, R_r_T_i, R_s_i, R_c_i)

        mse_loss, r_loss = self.Define_Loss(output_i, label_i, log_var_i)
        loss = mse_loss + reg

        # Keep original key ("val_Loss") but use actual batch size for logging
        self.log("val_Loss", r_loss, batch_size=label_i.size(0))
        return loss
