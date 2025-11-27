# models.py
import torch
import torch.nn as nn

# ---- CNN "local patch + viscosity" (notebook CNN version) ----
# CNN controller definition:
# INPUTS: patch (spatial neighbourhood of the field) and nu (the viscosity).
# shape: (x, y, z)
# x = batch_size (number of patches you process at once).
# y = no. of output feature maps produced.
# z = output sequence length.
class CNNController(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        # Two input channels (in_channels = 2): field patch and viscosity
        # First convolutional layer:
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=patch_size, padding=0) # output shape: (batch_size, 32, 1).
        # ReLU activation function -> introduces non-linearity
        self.activation1 = nn.ReLU()
        # Add batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(32)
        # Second convolutional layer: kernel_size = 1 since this layer only considers the current point and not neighbouring points.
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        # ReLU activation function -> introduces non-linearity
        self.activation2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(64)
        # Pooling layer -> reduce to single value
        self.pool = nn.AdaptiveMaxPool1d(1) # pooling reduces spatial dimension to 1 before fully connected
        # Connected layer:
        self.final_fc = nn.Sequential(
            nn.Flatten(start_dim=1), # flattens feature map to size (batch_size, 32)
            nn.Linear(64, 64), # maps all 64 features into 64 outputs
            # the predicted value at the next time step at the center of the patch.
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1) # maps all 64 features into a single output.
        )

    def forward(self, patch, nu):
        # patch: (batch_size, patch_size)
        # nu: (batch_size, 1)
        patch = patch.unsqueeze(1) # (batch_size, 1, patch_size)
        nu_channel = nu.unsqueeze(2).expand(-1, -1, patch.shape[2]) # (batch_size, 2, patch_size)
        
        # Concatenate channels
        x = torch.cat([patch, nu_channel], dim=1)  # x shape: (batch_size, 2, patch_size)
        # First convolutional layer: takes 2 input channels and outputs 16 feature maps.
        x = self.conv1(x)  # (batch_size, 32, 1)
        x = self.bn1(x)
        # ReLU activation function:
        x = self.activation1(x)
        # Second convolutional layer: takes 32 input channels and outputs 32 output channels.
        x = self.conv2(x)  # (batch_size, 64, 1)
        x = self.bn2(x)
        # ReLU activation function:
        x = self.activation2(x)
        # Reduce each filter's output to a single value using pooling layer:
        x = self.pool(x)  # (batch_size, 64, 1)
        x = x.view(x.shape[0], -1) # (batch_size, 64)
        x = self.final_fc(x)  # (batch_size, 1)
        return x
    

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





    class RNNController(nn.Module):
        def __init__(self, patch_size, hidden_size=64, rnn_type='LSTM', num_layers = 2):
            super().__init__()
            self.patch_size = patch_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers


            input_size = patch_size + 1  # patch values (features) + viscosity
            
            if rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                )
            elif rnn_type == 'GRU':
                self.rnn = nn.GRU(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                    )
            else:
                print("RNN type must be 'LSTM' or 'GRU'")
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # output prediction for patch center
            )

        def forward(self, patch, nu):
            # patch: (batch_size, seq_len, patch_size)
            # nu: (batch_size, 1)

            # Expand nu to (batch_size, seq_len, 1) for concatenation
            nu_expanded = nu.unsqueeze(1).expand(-1, patch.size(1), -1)  # (batch_size, seq_len, 1)
            rnn_input = torch.cat([patch, nu_expanded], dim=-1) # shape: (batch_size, seq_len, patch_size + 1)
            # RNN forward pass
            output, _ = self.rnn(rnn_input)  # output: (batch_size, seq_len, hidden_size)
            
            # Use the last output (corresponding to center patch point) for prediction
            last_output = output[:, -1, :] # (batch_size, hidden_size)

            # Predict next center patch value
            pred = self.fc(last_output) # shape: (batch_size, 1)
            return pred.squeeze(1) # (batch_size,); batch_size = 1
    



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
    


class CNNSpaceTimeController(nn.Module):
    """
    2D CNN sur (temps, espace) + viscosité.
    Entrée:
      - patch_seq : (B, L, P)   avec L = chunk_size (nb d'itérations passées),
                                 P = patch_size (2*patch_radius+1)
      - nu        : (B, 1)
    Sortie:
      - y         : (B,) valeur prédite au centre du patch à t+L
    """
    def __init__(
        self,
        patch_size: int,
        hidden_channels: int = 64,
        kernel_t: int = 3,
        kernel_x: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size

        # padding "same" manuel pour être stable sur toutes les versions de PyTorch
        pad_t = kernel_t // 2
        pad_x = kernel_x // 2

        self.conv1 = nn.Conv2d(
            in_channels=2,  # champ + nu
            out_channels=hidden_channels,
            kernel_size=(kernel_t, kernel_x),
            padding=(pad_t, pad_x),
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),        # (B, hidden_channels)
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, patch_seq, nu):
        """
        patch_seq : (B, L, P)
        nu        : (B, 1)
        """
        B, L, P = patch_seq.shape

        # canal 1 : le champ (temps, espace)
        x_field = patch_seq.view(B, 1, L, P)  # (B, 1, L, P)

        # canal 2 : la viscosité broadcastée sur (L, P)
        nu_plane = nu.view(B, 1, 1, 1).expand(-1, 1, L, P)  # (B, 1, L, P)

        x = torch.cat([x_field, nu_plane], dim=1)  # (B, 2, L, P)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.pool(x)                 # (B, hidden_channels, 1, 1)
        x = self.fc(x)                   # (B, 1)
        return x.squeeze(1)              # (B,)
