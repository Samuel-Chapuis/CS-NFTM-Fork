# main_cnn.py
import torch
from pathlib import Path

from data_loading import create_simple_dataloader
from models import CNNControllerPatch
from training import train_cnn_patch
from visualization import plot_trajectories
from evaluation import psnr, ssim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Data loader "simple" pour CNN ---
    cur = Path(".").resolve()
    files_burger = {
        0.001: cur / ".." / "Data" / "burgers1D_training_data_Nu0.001.npz",
        0.01:  cur / ".." / "Data" / "burgers1D_training_data_Nu0.01.npz",
        0.1:   cur / ".." / "Data" / "burgers1D_training_data_Nu0.1.npz",
        0.5:   cur / ".." / "Data" / "burgers1D_training_data_Nu0.5.npz",
    }
    train_loader = create_simple_dataloader(files_burger, batch_size=4)

    # --- 2) Modèle CNN ---
    patch_radius = 1
    model = CNNControllerPatch(patch_size=2 * patch_radius + 1)

    # --- 3) Training loop ---
    train_losses = train_cnn_patch(
        model,
        dataloader=train_loader,
        device=device,
        num_epochs=10,
        patch_radius=patch_radius,
    )

    # --- 4) Show results ---
    # on prend un sample pour visualiser la trajectoire
    init_field, true_traj, nu = next(iter(train_loader))
    true_traj = true_traj[0].to(device)  # (T, N)

    model.eval()
    with torch.no_grad():
        T, N = true_traj.shape
        preds = []
        for t in range(T - 1):
            field_t = true_traj[t].unsqueeze(0)  # (1, N)
            patches = extract_spatial_patches(field_t, patch_radius)      # (1, N, P)
            patches_flat = patches.reshape(N, -1)                         # (N, P)
            nu_vals = torch.full((N, 1), float(nu[0]), device=device)     # (N, 1)
            pred_next = model(patches_flat, nu_vals)                      # (N,)
            preds.append(pred_next)
        pred_traj = torch.stack(preds, dim=0)                             # (T-1, N)
        pred_traj = torch.cat([true_traj[0:1, :], pred_traj], dim=0)      # (T, N)

    from training import extract_spatial_patches  # import local

    plot_trajectories(true_traj, pred_traj, title_suffix=f" (nu={float(nu[0]):.4f})")

    # --- 5) Evaluation ---
    p = psnr(true_traj, pred_traj, max_val=1.0)
    s = ssim(true_traj, pred_traj, val_range=1.0)
    print(f"PSNR={p:.3f} dB, SSIM={s:.3f}")


if __name__ == "__main__":
    main()
