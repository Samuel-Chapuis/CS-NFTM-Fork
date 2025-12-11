# models.py
import torch
import torch.nn as nn



# CNN Controller for NFTM
class CNNController(nn.Module):
    def __init__(self, field_size):
        super().__init__()
        self.field_size = field_size

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        # self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        # self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(32)
        # self.dropout3 = nn.Dropout(0.1)

        self.conv_out = nn.Conv1d(32, 1, kernel_size=5, padding=2)


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, field):
        B, N = field.shape
        x = field.unsqueeze(1)  # (B,1,N)
         
        x = torch.relu(self.bn1(self.conv1(x)))
        # x = self.dropout1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        # x = self.dropout2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        # x = self.dropout3(x)

        delta = self.conv_out(x).squeeze(1)  # (B, N)
        pred_next_field = field + delta
        return pred_next_field


class RNNControllerPatch(nn.Module):
    """
    Input: temporal patch (chunk_size, patch_size) + viscosity
    -> scalar prediction.
    patch: (B, seq_len, patch_size)
    nu   : (B, 1)
    """
    def _init_(self, patch_size: int, hidden_size: int = 64, rnn_type: str = "LSTM"):
        super()._init_()
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



class TransformerController(nn.Module):
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
    
# Backward compatibility alias: some notebooks and scripts referenced the old
# class name `CNNControllerHistory`. After a refactor this class was renamed
# to `TransformerController`. Keep an alias here so old imports keep working.
CNNControllerHistory = TransformerController


# class CNNSpaceTimeController(nn.Module):
#     """
#     2D CNN sur (temps, espace) + viscosité.
#     Entrée:
#       - patch_seq : (B, L, P)   avec L = chunk_size (nb d'itérations passées),
#                                  P = patch_size (2*patch_radius+1)
#       - nu        : (B, 1)
#     Sortie:
#       - y         : (B,) valeur prédite au centre du patch à t+L
#     """
#     def __init__(
#         self,
#         patch_size: int,
#         hidden_channels: int = 64,
#         kernel_t: int = 3,
#         kernel_x: int = 3,
#     ):
#         super().__init__()
#         self.patch_size = patch_size

#         # padding "same" manuel pour être stable sur toutes les versions de PyTorch
#         pad_t = kernel_t // 2
#         pad_x = kernel_x // 2

#         self.conv1 = nn.Conv2d(
#             in_channels=2,  # champ + nu
#             out_channels=hidden_channels,
#             kernel_size=(kernel_t, kernel_x),
#             padding=(pad_t, pad_x),
#         )
#         self.bn1 = nn.BatchNorm2d(hidden_channels)
#         self.act1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(
#             in_channels=hidden_channels,
#             out_channels=hidden_channels,
#             kernel_size=(3, 3),
#             padding=(1, 1),
#         )
#         self.bn2 = nn.BatchNorm2d(hidden_channels)
#         self.act2 = nn.ReLU()

#         self.pool = nn.AdaptiveMaxPool2d((1, 1))

#         self.fc = nn.Sequential(
#             nn.Flatten(start_dim=1),        # (B, hidden_channels)
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_channels, 1),
#         )

#     def forward(self, patch_seq, nu):
#         """
#         patch_seq : (B, L, P)
#         nu        : (B, 1)
#         """
#         B, L, P = patch_seq.shape

#         # canal 1 : le champ (temps, espace)
#         x_field = patch_seq.view(B, 1, L, P)  # (B, 1, L, P)

#         # canal 2 : la viscosité broadcastée sur (L, P)
#         nu_plane = nu.view(B, 1, 1, 1).expand(-1, 1, L, P)  # (B, 1, L, P)

#         x = torch.cat([x_field, nu_plane], dim=1)  # (B, 2, L, P)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.act1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.act2(x)

#         x = self.pool(x)                 # (B, hidden_channels, 1, 1)
#         x = self.fc(x)                   # (B, 1)
#         return x.squeeze(1)              # (B,)


# class SingleChannelSpaceTimeCNN(nn.Module):
#     """
#     CNN spatio-temporel avec 1 seul canal de sortie à chaque couche.
#     Toujours très explicite (peu de poids), mais non linéaire.
#     """
#     def __init__(self, patch_size: int, history_len: int):
#         super().__init__()
#         self.patch_size = patch_size
#         self.history_len = history_len

#         self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

#         self.act = nn.Tanh()  # ou ReLU, mais Tanh colle bien à des valeurs bornées
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, patch_seq, nu):
#         B, L, P = patch_seq.shape

#         x_field = patch_seq.view(B, 1, L, P)
#         nu_plane = nu.view(B, 1, 1, 1).expand(-1, 1, L, P)
#         x = torch.cat([x_field, nu_plane], dim=1)  # (B,2,L,P)

#         x = self.act(self.conv1(x))   # (B,1,L,P)
#         x = self.act(self.conv2(x))   # (B,1,L,P)
#         x = self.pool(x).view(B, 1)   # (B,1)
#         return x.squeeze(1)           # (B,)
