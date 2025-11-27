# training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from typing import Dict, List
from models import CNNControllerPatch, RNNControllerPatch, CNNControllerHistory, RNNController

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



# We pass as argument the batch of fields.
def extract_patches(fields_batch, patch_radius=1):
    # fields_batch shape: (batch_size, N)
    batch_size, N = fields_batch.shape
    x = fields_batch.unsqueeze(1)  # (batch_size, 1, N)
    pad = nn.ReplicationPad1d(patch_radius)
    padded = pad(x)  # (batch_size, 1, N + 2*patch_radius)
    patches = padded.unfold(dimension=2, size=2*patch_radius+1, step=1)  # (batch_size, 1, N, patch_size)
    patches = patches.squeeze(1)  # (batch_size, N, patch_size)
    return patches


# --------------- Helper functions for evaluation ----------

def plot_learning_progress(true_traj, pred_traj, epoch, sample_idx=0):
    true_np = true_traj.cpu().numpy().T  # (space, time)
    pred_np = pred_traj.cpu().numpy().T
    error_np = (true_np - pred_np) ** 2
    # error_np = abs(true_np - pred_np)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(true_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='u(x,t=0)')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title(f'True Trajectory (Sample {sample_idx})')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='u(x,t=0)')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title(f'Predicted Trajectory (Epoch {epoch})')

    plt.subplot(1, 3, 3)
    plt.imshow(error_np, aspect='auto', cmap='inferno')
    plt.colorbar(label='error')
    plt.xlabel('Time t')
    plt.ylabel('Position x')
    plt.title('Absolute Error')

    plt.tight_layout()
    plt.show()


def psnr(true, pred, max_val=1.0):
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val**2 / mse)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(true, pred, window_size=11, size_average=True, val_range=1.0):
    # Assume inputs are 2D tensors (Height=space, Width=time) or batch versions
    # Convert to 4D tensor [N,C,H,W] for conv2d
    if true.dim() == 2:
        true = true.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

    channel = true.size(1)
    window = create_window(window_size, channel).to(true.device)

    mu1 = torch.nn.functional.conv2d(true, window, padding=window_size//2, groups=channel)
    mu2 = torch.nn.functional.conv2d(pred, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(true * true, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(true * pred, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item() if size_average else ssim_map


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




# ---------- Training loop for RNN ----------

def train_rnn(
    model: RNNController,
    trainDataloader,
    testDataloader,
    device: torch.device,
    num_epochs: int = 30,
    roll_out_size: int = 10,  # number of steps to roll out
    patch_radius: int = 1,
):
    """
    Loop inspired by RNN_burgers_1D.py:
    - 
    """
    model.to(device)
    model.train()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    patch_size = 2 * patch_radius + 1

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # we iterate over training data, which consists of 3 trajectories (one per viscosity), since only have 1 initial conditon per viscosity.
        for initial_fields_batch, true_trajectories_batch, viscosities_batch in trainDataloader:
            initial_fields_batch = initial_fields_batch.to(device) # (batch_size, N); batch_size=1
            true_trajectories_batch = true_trajectories_batch.to(device) # (batch_size, T, N); batch_size=1
            viscosities_batch = viscosities_batch.to(device) # (batch_size, 1); batch_size=1

            _, T, N = true_trajectories_batch.shape
            total_loss = 0.0
            num_rollouts = 0
            
            for t in range(0, T - roll_out_size, 5): # don't go through all time steps, just every 5 steps
                # start from true f_t (at the very beginning, f_0, then, f_5, f_10,...)
                current_field = true_trajectories_batch[:, t, :]  # (1, N)
                # autoregressive rollout: predict f_{t+1},...,f_{t+roll_out_size}
                for roll_step in range(roll_out_size):
                    patches = extract_patches(current_field, patch_radius=1)  # (batch_size, N, patch_size); batch_size=1
                    # we need (batch_patches, seq_len=1, patch_size) for RNN
                    patches_reshape = patches.reshape(-1, 1, patches.shape[-1])  # (N, 1, patch_size)
                    # expand viscosity for each patch: (N, 1)
                    viscosity_expanded = viscosities_batch.expand(patches_reshape.shape[0], -1) # (N, 1)
                    # RNN forward: predicts center value for each patch
                    next_pred = model(patches_reshape, viscosity_expanded)  # (N,) after squeeze in forward.
                    next_pred = next_pred.view(1, N)  # field at next time step: (1, N)
                    
                    # use prediction as next input (autoregressive)
                    current_field = next_pred

                # We are at t = t + roll_out_size
                
                # when reach roll_out_size (roll_step = roll_out_size - 1), stop and collect predictions
                prediction_at_rollout = current_field # f_{t+roll_out_size}, shape: (1, N)
                true_at_rollout = true_trajectories_batch[:, t + roll_out_size, :] # (1, N)

                # Compute loss at rollout step:
                rollout_loss = mse_loss(prediction_at_rollout, true_at_rollout)
                total_loss += rollout_loss # accumulate loss over rollouts
                num_rollouts += 1

            # We finished all rollouts for this trajectory.

            # We train on the average loss over all rollouts for current trajectory:
            loss = total_loss / num_rollouts   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accumulates loss over each viscosity trajectory for the current epoch.
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss) # stores the average loss for the current epoch
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # EVALUATION/TESTING LOOP (every 10 epochs):
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for sample_idx, (initial_field, true_trajectory, viscosity_val) in enumerate(testDataloader):
                # sample_idx = 0  # fixed index to visualize progress
                # initial_field, true_trajectory, viscosity_val = dataset[sample_idx]
                    # shapes: (1, N), (1, T, N), (1, 1)
                    initial_field = initial_field.to(device)
                    true_trajectory = true_trajectory.to(device)
                    viscosity_val = viscosity_val.to(device)
                    T, N = true_trajectory.shape[1], true_trajectory.shape[2]

                    pred_trajectory = []
                    for t in range(T - 1):
                        current_field = true_trajectory[:,t,:]  # (1, N)
                        patches = extract_patches(current_field, patch_radius=1) # (1, N, patch_size)
                        patches_reshape = patches.reshape(-1, 1, patches.shape[-1])  # (N, 1, patch_size)
                        # nu_vals = torch.full((patches_reshape.shape[0], 1), float(viscosity_val), device=device)
                        nu_vals = viscosity_val.expand(patches_reshape.shape[0], -1) # (N, 1)
                        next_pred = model(patches_reshape, nu_vals) # (N,)
                        next_pred = next_pred.view(1,N) # next predicted field at next time step: (1, N)
                        pred_trajectory.append(next_pred)
            
                    pred_trajectory = torch.stack(pred_trajectory, dim=0) # (T-1, 1, N)
                    pred_trajectory_2d = pred_trajectory.squeeze(1)  # # (T-1, 1, N) -> (T-1, N)
                    init_2d = initial_field.squeeze(0).unsqueeze(0) # (1, N)
                    pred_trajectory_2d = torch.cat([init_2d, pred_trajectory_2d], dim=0) # (T, N)
                    true_trajectory_2d = true_trajectory.squeeze(0) # (1, T, N) -> (T, N)
                    # Visualize for first test sample only
                    if sample_idx == 0:
                        print("="*50)
                        print("Visualization of learning progress at epoch:", epoch + 1)
                        plot_learning_progress(true_trajectory_2d, pred_trajectory_2d, epoch + 1)
                        print("PSNR and SSIM metrics at epoch:", epoch + 1)
                        psnr_val = psnr(true_trajectory_2d, pred_trajectory_2d, max_val=1.0)
                        ssim_val = ssim(true_trajectory_2d, pred_trajectory_2d, val_range=1.0)
                        print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                        print("="*50 + "\n")
            model.train()  # back to training mode
        
    return epoch_losses



# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


