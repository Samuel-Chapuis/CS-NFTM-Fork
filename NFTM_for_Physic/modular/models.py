# models.py
import torch
import torch.nn as nn

# ---- CNN "local patch + viscosity" (version notebook CNN) ----

class CNNControllerPatch(nn.Module):
    """
    Patch 1D + viscosité -> valeur prédite au centre
    patch: (B, patch_size)
    nu   : (B, 1)
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=patch_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, patch, nu):
        # patch: (B, P), nu: (B, 1)
        patch = patch.unsqueeze(1)                 # (B, 1, P)
        nu_channel = nu.unsqueeze(2).expand(-1, 1, patch.size(2))  # (B, 1, P)
        x = torch.cat([patch, nu_channel], dim=1)  # (B, 2, P)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)
        x = self.fc(x)
        return x.squeeze(1)                        # (B,)


# ---- RNN controller (version notebook RNN) ----

class RNNControllerPatch(nn.Module):
    """
    Input: patch temporel (chunk_size, patch_size) + viscosité
    -> prédiction scalaire.
    patch: (B, seq_len, patch_size)
    nu   : (B, 1)
    """
    def __init__(self, patch_size: int, hidden_size: int = 64, rnn_type: str = "LSTM"):
        super().__init__()
        input_size = patch_size + 1  # patch + nu
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, patch_seq, nu):
        # patch_seq: (B, seq_len, patch_size)
        # nu      : (B, 1)
        nu_expanded = nu.unsqueeze(1).expand(-1, patch_seq.size(1), -1)  # (B, seq_len, 1)
        x = torch.cat([patch_seq, nu_expanded], dim=-1)                  # (B, seq_len, patch_size+1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]                                             # (B, hidden)
        y = self.fc(last)                                                # (B, 1)
        return y.squeeze(1)


# ---- CNN "HISTORY_LEN channels" (version sam_cnn) optionnelle ----

class CNNControllerHistory(nn.Module):
    """
    CNN de sam_cnn.py : input (B*N, L, patch_size) avec L = HISTORY_LEN canaux.
    """
    def __init__(self, history_len: int, patch_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=history_len, out_channels=8, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.spatial_reduce = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x: (B*N, L, patch_size)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.spatial_reduce(x)
        x = self.output_layer(x)  # (B*N, 1, 1)
        return x.view(x.size(0))  # (B*N,)
