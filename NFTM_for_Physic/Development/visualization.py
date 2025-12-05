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


def plot_space_time_kernel(
    model,
    layer_name: str = "conv1",
    channel_type: str = "field",
    out_channel: int | None = None,
    cmap: str = "bwr",
):
    """
    Visualise le noyau spatio-temporel d'une couche conv2d (par défaut `conv1`)
    pour des modèles comme:
      - SingleChannelSpaceTimeCNN (conv1 : 2 -> 1, kernel 3x3)
      - CNNSpaceTimeController    (conv1 : 2 -> C, kernel k_t x k_x)

    - model        : modèle PyTorch (doit avoir un attribut layer_name)
    - layer_name   : nom de la couche, ex. "conv1" ou "conv2"
    - channel_type : "field" (u) ou "nu" (viscosité)
    - out_channel  :
        * None -> moyenne sur tous les filtres de sortie
        * int  -> un filtre de sortie spécifique
    """

    if not hasattr(model, layer_name):
        print(
            f"plot_space_time_kernel: le modèle n'a pas de couche '{layer_name}'. "
            "Rien à afficher."
        )
        return

    layer = getattr(model, layer_name)
    if not hasattr(layer, "weight"):
        print(
            f"plot_space_time_kernel: la couche '{layer_name}' n'a pas d'attribut 'weight'. "
            "Rien à afficher."
        )
        return

    weight = layer.weight.detach().cpu().numpy()
    # On ne traite que les conv 2D: (C_out, C_in, k_t, k_x)
    if weight.ndim != 4:
        print(
            f"plot_space_time_kernel: '{layer_name}.weight' a une shape {weight.shape}, "
            "attendu: 4D (C_out, C_in, k_t, k_x). Modèle ignoré."
        )
        return

    C_out, C_in, k_t, k_x = weight.shape

    # Sélection du canal d'entrée: champ ou viscosité
    if channel_type == "field":
        in_idx = 0
    elif channel_type == "nu":
        if C_in < 2:
            print(
                f"plot_space_time_kernel: channel_type='nu' demandé mais la couche "
                f"'{layer_name}' n'a que {C_in} canal(x) d'entrée. "
                "On affiche le canal 0 à la place."
            )
            in_idx = 0
        else:
            in_idx = 1
    else:
        raise ValueError("channel_type doit être 'field' ou 'nu'.")

    if in_idx >= C_in:
        print(
            f"plot_space_time_kernel: index de canal d'entrée {in_idx} "
            f">= C_in={C_in}. Rien à afficher."
        )
        return

    # Sélection du filtre de sortie
    if out_channel is None:
        # moyenne des noyaux sur tous les filtres de sortie
        kernel = weight[:, in_idx, :, :].mean(axis=0)  # (k_t, k_x)
        title = (
            f"{layer_name} – {channel_type} kernel "
            f"(mean over {C_out} filters)"
        )
    else:
        if not (0 <= out_channel < C_out):
            print(
                f"plot_space_time_kernel: out_channel={out_channel} hors bornes "
                f"[0, {C_out-1}]. Rien à afficher."
            )
            return
        kernel = weight[out_channel, in_idx, :, :]      # (k_t, k_x)
        title = (
            f"{layer_name} – {channel_type} kernel – filter {out_channel}"
        )

    # Plot
    t_offsets = np.arange(k_t) - k_t // 2
    x_offsets = np.arange(k_x) - k_x // 2

    plt.figure(figsize=(5, 4))
    plt.imshow(kernel, cmap=cmap, aspect="auto", origin="lower")
    plt.colorbar(label="weight")
    plt.xticks(np.arange(k_x), x_offsets)
    plt.yticks(np.arange(k_t), t_offsets)
    plt.xlabel("space offset (Δx)")
    plt.ylabel("time offset (Δt)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
