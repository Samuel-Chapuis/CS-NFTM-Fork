# training.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from models import CNNControllerPatch, RNNControllerPatch, CNNControllerHistory

# ---------- Patch extraction helpers ----------

def extract_spatial_patches(field_batch, patch_radius: int = 1):
    """
    Simple CNN version:
    field_batch: (B, N)
    -> patches: (B, N, patch_size)
    """
    B, N = field_batch.shape
    x = field_batch.unsqueeze(1)  # (B, 1, N)
    pad = nn.ReplicationPad1d(patch_radius)
    padded = pad(x)               # (B, 1, N+2r)
    patches = padded.unfold(2, 2 * patch_radius + 1, 1)  # (B, 1, N, P)
    return patches.squeeze(1)     # (B, N, P)


def build_patches_from_sequence(fields_seq, r: int, patch_size: int):
    """
    sam_cnn version:
    fields_seq: (B, L, N) -> (B*N, L, patch_size)
    """
    B, L, N = fields_seq.shape
    patches_list = []
    for l in range(L):
        field_l = fields_seq[:, l, :]                       # (B, N)
        padded_l = F.pad(field_l, (r, r), mode='replicate') # (B, N+2r)
        patches_l = padded_l.unfold(1, patch_size, 1)       # (B, N, P)
        patches_list.append(patches_l)
    patches_seq = torch.stack(patches_list, dim=2)          # (B, N, L, P)
    return patches_seq.reshape(B * N, L, patch_size)

# ---------- Training loop for "simple" CNN ----------

def train_cnn_patch(
    model: CNNControllerPatch,
    dataloader,
    device: torch.device,
    num_epochs: int = 30,
    patch_radius: int = 1,
):
    """
    CNN training cnn.py style:
    - We use ground truth at t as input (no true auto-regressive rollout here).
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    patch_size = 2 * patch_radius + 1

    epoch_losses: List[float] = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for init_field, true_traj, nu in dataloader:
            true_traj = true_traj.to(device)  # (B, T, N)
            nu = nu.to(device)                # (B, 1)
            B, T, N = true_traj.shape

            all_preds = []
            for t in range(T - 1):
                current_field = true_traj[:, t, :]      # (B, N)
                patches = extract_spatial_patches(current_field, patch_radius)  # (B, N, P)
                patches_flat = patches.reshape(B * N, patch_size)              # (B*N, P)
                nu_expanded = nu.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 1)  # (B*N,1)
                pred_next = model(patches_flat, nu_expanded).reshape(B, N)    # (B,N)
                all_preds.append(pred_next)

            pred_traj = torch.stack(all_preds, dim=1)       # (B, T-1, N)
            target_traj = true_traj[:, 1:, :]               # (B, T-1, N)

            loss = criterion(pred_traj, target_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"[CNN] Epoch {epoch+1}/{num_epochs} - MSE: {avg_loss:.6e}")

    return epoch_losses


# ---------- Training loop for RNN ----------

def train_rnn_patch(
    model: RNNControllerPatch,
    dataloader,
    device: torch.device,
    chunk_size: int = 3,
    num_epochs: int = 30,
    patch_radius: int = 1,
):
    """
    Loop inspired by rnn.py:
    - we take temporal chunks (chunk_size) as input
    - target: field at time t+chunk_size
    """
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    patch_size = 2 * patch_radius + 1

    epoch_losses: List[float] = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for init_field, true_traj, nu in dataloader:
            true_traj = true_traj.to(device)  # (B, T, N)
            nu = nu.to(device)                # (B, 1)
            B, T, N = true_traj.shape

            all_preds = []
            all_targets = []
            for t in range(T - chunk_size):
                current_chunk = true_traj[:, t : t + chunk_size, :]    # (B, chunk_size, N)
                next_true = true_traj[:, t + chunk_size, :]            # (B, N)

                # patches: (B, chunk_size, N, P) -> (B*N, chunk_size, P)
                patches = build_patches_from_sequence(current_chunk, patch_radius, patch_size)
                # patches: (B*N, chunk_size, P)
                nu_expanded = nu.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 1)  # (B*N,1)
                pred_next = model(patches, nu_expanded).reshape(B, N)              # (B,N)

                all_preds.append(pred_next)
                all_targets.append(next_true)

            if not all_preds:
                continue
            pred_traj = torch.stack(all_preds, dim=1)      # (B, T', N)
            target_traj = torch.stack(all_targets, dim=1)  # (B, T', N)

            loss = criterion(pred_traj, target_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"[RNN] Epoch {epoch+1}/{num_epochs} - MSE: {avg_loss:.6e}")

    return epoch_losses


# ---------- Training loop for CNN SpaceTimeController ----------

