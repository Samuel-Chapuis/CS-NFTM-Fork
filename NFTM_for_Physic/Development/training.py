# training.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from models import  CNNController, TransformerController, RNNControllerPatch


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
    return patches.squeeze(1) # (B, N, P)


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



# ------------ Recurrent CNN training loop -----------------------------

def train_cnn_controller(cnn_controller, train_loader, test_loader, num_epochs=30, roll_out_size=10, device='cuda', plot_learning_progress=None):
    """
    Complete training and evaluation loop for CNN controller on viscosity trajectories.
    
    Args:
        cnn_controller: PyTorch CNN model
        train_loader: DataLoader with (initial_fields_batch, true_trajectories_batch, viscosities_batch)
        test_loader: DataLoader for evaluation with (initial_field, true_trajectory, viscosity_val)
        optimizer: PyTorch optimizer (e.g., torch.optim.Adam(cnn_controller.parameters()))
        mse_loss: MSE loss function (nn.MSELoss())
        num_epochs: Number of training epochs [default: 30]
        roll_out_size: Number of autoregressive rollout steps [default: 10]
        device: torch device ('cuda' or 'cpu') [default: 'cuda']
        plot_learning_progress: Optional function for visualization (true_traj, pred_traj, epoch)
    
    Returns:
        epoch_losses: List of average losses per epoch

    """
    epoch_losses = []
    mse_loss = nn.MSELoss() # LOSS FUNCTION
    optimizer = torch.optim.Adam(cnn_controller.parameters(), lr=1e-4, weight_decay=1e-4) # OPTIMIZER
    for epoch in range(num_epochs):
        cnn_controller.train()
        epoch_loss = 0.0

        # Training: iterate over training data (viscosity trajectories)
        for initial_fields_batch, true_trajectories_batch, viscosities_batch in train_loader:
            initial_fields_batch = initial_fields_batch.to(device)  # (B, N)
            true_trajectories_batch = true_trajectories_batch.to(device)  # (B, T, N)
            
            _, T, N = true_trajectories_batch.shape
            total_loss = 0.0
            num_rollouts = 0
            
            # Multiple rollouts per trajectory (every 10 steps)
            for t in range(0, T - roll_out_size, 10):
                # Start from true field f_t
                current_field = true_trajectories_batch[:, t, :]  # (B, N)
                
                # Autoregressive rollout: f_{t+1}, ..., f_{t+roll_out_size}
                for roll_step in range(roll_out_size):
                    next_field = cnn_controller(current_field)  # (B, N)
                    current_field = next_field  # Recurrent update
                
                # Compare prediction at rollout endpoint
                prediction_at_rollout = current_field  # f_{t+roll_out_size}
                true_at_rollout = true_trajectories_batch[:, t + roll_out_size, :]
                rollout_loss = mse_loss(prediction_at_rollout, true_at_rollout)
                total_loss += rollout_loss
                num_rollouts += 1

            # Average loss over rollouts for this batch, then backprop
            loss = total_loss / num_rollouts
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_controller.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # Evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_cnn_model(cnn_controller, test_loader, device, epoch, plot_learning_progress)
        
        cnn_controller.train()  # Ensure training mode
    
    return epoch_losses


def evaluate_cnn_model(cnn_controller, test_loader, device, epoch, plot_learning_progress=None):
    """Internal evaluation with 1-step MSE and full trajectory metrics."""
    cnn_controller.eval()
    with torch.no_grad():
        for sample_idx, (initial_field, true_trajectory, viscosity_val) in enumerate(test_loader):
            initial_field = initial_field.to(device)  # (1, N)
            true_trajectory = true_trajectory.to(device)  # (1, T, N)
            
            T_test, N_test = true_trajectory.shape[1], true_trajectory.shape[2]

            # 1-step prediction MSE
            f0_true = true_trajectory[:, 0, :]
            f1_true = true_trajectory[:, 1, :]
            f1_pred = cnn_controller(f0_true)
            mse_1step = torch.mean((f1_true - f1_pred)**2).item()
            print(f"[Epoch {epoch+1}] 1-step MSE: {mse_1step:.4e}")

            # Full autoregressive trajectory prediction
            pred_trajectory = []
            current_field = initial_field
            for t in range(T_test - 1):
                next_field = cnn_controller(current_field)
                pred_trajectory.append(next_field)
                current_field = next_field
            
            # Stack and align shapes for comparison: (T, N)
            pred_trajectory = torch.stack(pred_trajectory, dim=0).squeeze(1)  # (T-1, N)
            init_2d = initial_field.squeeze(0).unsqueeze(0)  # (1, N)
            pred_trajectory_2d = torch.cat([init_2d, pred_trajectory], dim=0)  # (T, N)
            true_trajectory_2d = true_trajectory.squeeze(0)  # (T, N)

            # Visualize first test sample only
            if sample_idx == 0:
                print("="*50)
                print(f"Visualization of learning progress at epoch: {epoch + 1}")
                
                mse_val = torch.mean((true_trajectory_2d - pred_trajectory_2d)**2).item()
                print(f"[Epoch {epoch+1}] global MSE on test traj: {mse_val:.4e}")
                
                if plot_learning_progress is not None:
                    plot_learning_progress(true_trajectory_2d, pred_trajectory_2d, epoch + 1)
                
                # Optional quality metrics (requires psnr/ssim functions)
                # psnr_val = psnr(true_trajectory_2d, pred_trajectory_2d, max_val=1.0)
                # ssim_val = ssim(true_trajectory_2d, pred_trajectory_2d, val_range=1.0)
                # print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                
                print("="*50 + "\n")
    
    cnn_controller.train()







