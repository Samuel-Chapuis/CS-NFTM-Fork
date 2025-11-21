# models.py
import torch
import torch.nn as nn

# ---- CNN "local patch + viscosity" (notebook CNN version) ----

class CNNControllerPatch(nn.Module):
    """
    1D Patch + viscosity -> predicted value at center
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


# ---- RNN controller (notebook RNN version) ----

class RNNControllerPatch(nn.Module):
    """
    Input: temporal patch (chunk_size, patch_size) + viscosity
    -> scalar prediction.
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


class CNNControllerHistory(nn.Module):
    """
    Input: temporal patch sequence + viscosity
    patch_seq: (B, seq_len, patch_size)
    nu       : (B, 1)
    -> scalar prediction with learned temporal weights (attention).
    """
    def __init__(self, patch_size: int, hidden_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # On encode chaque (patch + nu) en un embedding
        self.embed = nn.Sequential(
            nn.Linear(patch_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Score d'attention pour chaque time step
        self.attn = nn.Linear(hidden_size, 1)

        # Tête finale pour prédire le scalaire à partir du contexte agrégé
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, patch_seq, nu):
        # patch_seq: (B, seq_len, patch_size)
        # nu      : (B, 1)
        B, L, P = patch_seq.shape
        assert P == self.patch_size

        # On répète nu sur la dimension temporelle
        nu_expanded = nu.unsqueeze(1).expand(-1, L, 1)      # (B, L, 1)
        x = torch.cat([patch_seq, nu_expanded], dim=-1)     # (B, L, P+1)

        # Encodage pas de temps par pas de temps
        h = self.embed(x)                                   # (B, L, hidden)

        # Scores d'attention pour chaque time step
        scores = self.attn(h).squeeze(-1)                   # (B, L)
        weights = torch.softmax(scores, dim=-1)             # (B, L)

        # Combinaison pondérée des embeddings temporels
        context = (h * weights.unsqueeze(-1)).sum(dim=1)    # (B, hidden)

        # Prédiction finale
        y = self.fc_out(context)                            # (B, 1)
        return y.squeeze(1)                                 # (B,)
