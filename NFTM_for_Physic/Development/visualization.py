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


def plot_space_time_kernel(model,
                           channel_type: str = "field",
                           out_channel: int | None = None,
                           cmap: str = "bwr"):
    """
    Visualise le noyau spatio-temporel de la première couche conv (conv1)
    d'un modèle type CNNSpaceTimeController.

    - model : modèle entraîné, avec un attribut .conv1
    - channel_type : "field" (u) ou "nu" (viscosité)
    - out_channel :
        * None  -> moyenne des noyaux sur tous les filtres de sortie
        * int   -> on affiche un filtre de sortie spécifique
    """

    if not hasattr(model, "conv1"):
        raise ValueError("Le modèle fourni n'a pas d'attribut 'conv1'.")

    weight = model.conv1.weight.detach().cpu().numpy()   # (C_out, C_in, k_t, k_x)
    C_out, C_in, k_t, k_x = weight.shape

    if channel_type == "field":
        in_idx = 0
    elif channel_type == "nu":
        in_idx = 1
    else:
        raise ValueError("channel_type doit être 'field' ou 'nu'.")

    if in_idx >= C_in:
        raise ValueError(
            f"Le canal d'entrée index {in_idx} n'existe pas dans conv1 "
            f"(C_in={C_in}). Vérifie l'architecture."
        )

    if out_channel is None:
        # moyenne sur les filtres de sortie -> vue globale du schéma numérique appris
        kernel = weight[:, in_idx, :, :].mean(axis=0)    # (k_t, k_x)
        title = f"Conv1 {channel_type} kernel (mean over {C_out} filters)"
    else:
        if not (0 <= out_channel < C_out):
            raise ValueError(f"out_channel doit être dans [0, {C_out-1}]")
        kernel = weight[out_channel, in_idx, :, :]       # (k_t, k_x)
        title = f"Conv1 {channel_type} kernel – filter {out_channel}"

    plt.figure(figsize=(5, 4))
    plt.imshow(kernel, cmap=cmap, aspect="auto", origin="lower")
    plt.colorbar(label="weight")

    # Axes = offsets temporels et spatiaux autour du centre
    t_offsets = np.arange(k_t) - k_t // 2
    x_offsets = np.arange(k_x) - k_x // 2
    plt.xticks(np.arange(k_x), x_offsets)
    plt.yticks(np.arange(k_t), t_offsets)

    plt.xlabel("space offset (Δx)")
    plt.ylabel("time offset (Δt)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
