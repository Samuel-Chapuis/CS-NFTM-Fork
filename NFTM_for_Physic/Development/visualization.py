# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_trajectories(true_traj: torch.Tensor, pred_traj: torch.Tensor, title_suffix: str = ""):
    """
    true_traj, pred_traj : (T, N) (single trajectory)
    """
    true_np = true_traj.cpu().numpy().T   # (N, T)
    pred_np = pred_traj.cpu().numpy().T
    err_np  = (true_np - pred_np) ** 2

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(true_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("True" + title_suffix)

    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("Predicted" + title_suffix)

    plt.subplot(1, 3, 3)
    plt.imshow(err_np, aspect="auto", cmap="inferno")
    plt.colorbar(label="squared error")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title("Error")

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, test_losses=None, title: str = "Loss per epoch"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker="o", label="train")
    if test_losses is not None:
        plt.plot(test_losses, marker="s", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
