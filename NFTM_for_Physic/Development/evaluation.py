# evaluation.py
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training import extract_spatial_patches

def psnr(true: torch.Tensor, pred: torch.Tensor, max_val: float = 1.0) -> float:
    """
    true, pred : mêmes shapes (par ex. (T, N) ou (B, T, N))
    """
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse).item()


def _gaussian(window_size, sigma):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel=1):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(true: torch.Tensor, pred: torch.Tensor, window_size=11, val_range=1.0) -> float:
    """
    SSIM 2D (pour cartes (space,time)).
    true, pred : (H, W) ou (B,1,H,W)
    """
    if true.dim() == 2:
        true = true.unsqueeze(0).unsqueeze(0)
        pred = pred.unsqueeze(0).unsqueeze(0)

    channel = true.size(1)
    window = _create_window(window_size, channel).to(true.device)

    mu1 = F.conv2d(true, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(true * true, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(true * pred, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    R² global.
    """
    ssr = torch.sum((target - pred) ** 2)
    mean_target = torch.mean(target)
    sst = torch.sum((target - mean_target) ** 2) + eps
    r2 = 1 - ssr / sst
    return r2.item()


def get_loader_nu_values(loader):
    """Return a sorted list of unique nu values found in a DataLoader or dataset.

    The loader can be a DataLoader or any iterable returning (init, traj, nu).
    """
    nus = set()
    for batch in loader:
        try:
            nu = batch[2]
        except Exception:
            _, _, nu = batch

        if isinstance(nu, torch.Tensor):
            for v in nu.reshape(-1).cpu().numpy():
                try:
                    nus.add(float(v))
                except Exception:
                    continue
        else:
            try:
                nus.add(float(nu))
            except Exception:
                continue
    return sorted(nus)


def resolve_nu_target(
    loader,
    plot_nu=None,
    selected_nu=None,
    list_nus: bool = False,
    verbose: bool = True,
):
    """Utility to resolve a nu target from loader and a plot specifier.

    Arguments:
      loader: data loader or iterable used to enumerate available nu values
      plot_nu: one of None | float | 'min'|'max'|'idx:N'|'random'|'list'|str->float
      selected_nu: if provided, priority target (returned directly)
      list_nus: if True -> print the list of nus and return None
      verbose: prints helpful messages

    Returns:
      float | None
    """
    nus = []
    try:
        nus = get_loader_nu_values(loader)
    except Exception:
        if verbose:
            print("resolve_nu_target: unable to enumerate nus from loader")

    if list_nus and verbose:
        print(f"Loader contains {len(nus)} unique nu values: {nus}")

    if selected_nu is not None:
        if verbose:
            print(f"Using provided selected_nu: {selected_nu}")
        return float(selected_nu)

    if plot_nu is None:
        return None

    # numeric types: choose closest if available
    if isinstance(plot_nu, (int, float)):
        if nus:
            val = float(plot_nu)
            closest = min(nus, key=lambda x: abs(x - val))
            if abs(closest - val) > 1e-8 and verbose:
                print(f"Requested nu {val} not in loader; using nearest available {closest}.")
            return float(closest)
        else:
            return float(plot_nu)

    if isinstance(plot_nu, str):
        s = plot_nu.strip().lower()
        if s == 'list':
            # already printed nus above
            return None
        if s == 'min' and nus:
            return float(min(nus))
        if s == 'max' and nus:
            return float(max(nus))
        if s.startswith('idx:') and nus:
            try:
                idx = int(s.split(':', 1)[1])
                if 0 <= idx < len(nus):
                    return float(nus[idx])
                else:
                    if verbose:
                        print(f"resolve_nu_target: index {idx} out of range (0..{len(nus)-1})")
                    return None
            except Exception:
                if verbose:
                    print("resolve_nu_target: invalid idx format; use 'idx:N'")
                return None
        if s == 'random' and nus:
            import random
            return float(random.choice(nus))
        # fallback: try parse numeric string
        try:
            val = float(s)
            if nus:
                closest = min(nus, key=lambda x: abs(x - val))
                if abs(closest - val) > 1e-8 and verbose:
                    print(f"Requested nu {val} not in loader; using nearest available {closest}.")
                return float(closest)
            else:
                return float(val)
        except Exception:
            if verbose:
                print(f"resolve_nu_target: could not parse PLOT_NU string '{plot_nu}'")
            return None


# def generate_model_predictions(model, train_loader, device, patch_radius,
#                                verbose: bool = True, chunk_size: int = 3):
#     """
#     Génère une trajectoire prédite à partir d'un batch du train_loader.

#     Compatible avec :
#       - CNNControllerPatch (mode "1 time step")
#       - RNNControllerPatch (mode séquentiel)
#       - TransformerController (mode séquentiel avec contexte)
#       - CNNSpaceTimeController (mode séquentiel spatio-temporel)
#     """
#     from training import build_patches_from_sequence, extract_spatial_patches

#     # --- On récupère un batch et on ne garde que le premier exemple ---
#     init_field, true_traj, nu = next(iter(train_loader))
#     true_traj = true_traj[0].to(device)          # (T, N)
#     nu_value = float(nu[0].item())
#     T, N = true_traj.shape

#     if verbose:
#         print(f"Generating predictions for nu = {nu_value:.4f}")
#         print(f"True trajectory shape: {true_traj.shape}")
#         print(f"Model type: {type(model).__name__}")

#     model.eval()
#     patch_size = 2 * patch_radius + 1

#     # --- Détection par nom de classe, pas par isinstance ---
#     model_name = type(model).__name__

#     # CNN 1-step : patch spatial (B, P) + nu
#     is_cnn_patch = model_name in ["CNNControllerPatch", "CNNController"]

#     # Tout le reste = modèles séquentiels (RNN, CNNHistory, CNNSpaceTime, Transformer…)
#     is_seq_model = not is_cnn_patch

#     preds = []

#     with torch.no_grad():

#         # ------------------------------------------------------
#         # 1) MODE SÉQUENTIEL : RNN / CNNHistory / SpaceTimeCNN
#         # ------------------------------------------------------
#         if is_seq_model:
#             if T <= chunk_size:
#                 if verbose:
#                     print(f"Warning: T={T} <= chunk_size={chunk_size}, "
#                           "on renvoie la vérité terrain comme prédiction.")
#                 pred_traj = true_traj.clone()
#             else:
#                 if verbose:
#                     print(f"Sequential mode: {T - chunk_size} prédictions "
#                           f"avec chunk_size={chunk_size}")

#                 for t in range(T - chunk_size):
#                     if verbose and t % 10 == 0:
#                         print(f"  step {t}/{T - chunk_size}")

#                     # Historique temporel : (1, chunk_size, N)
#                     current_chunk = true_traj[t:t + chunk_size, :].unsqueeze(0)

#                     # Patches spatio-temporels : (N, chunk_size, patch_size)
#                     patches = build_patches_from_sequence(
#                         current_chunk, patch_radius, patch_size
#                     )

#                     # Nu pour chaque patch spatial
#                     nu_vals = torch.full(
#                         (patches.size(0), 1),
#                         nu_value,
#                         device=device
#                     )  # (N, 1)

#                     # Prédiction du pas de temps suivant pour chaque point spatial
#                     pred_next = model(patches, nu_vals)  # (N,)
#                     preds.append(pred_next)

#                 # (T - chunk_size, N)
#                 pred_future = torch.stack(preds, dim=0)

#                 # On recolle les chunk_size premiers instants en vérité terrain
#                 pred_traj = torch.cat(
#                     [true_traj[:chunk_size, :], pred_future],
#                     dim=0
#                 )  # (T, N)

#         # ------------------------------------------------------
#         # 2) MODE CNN 1-STEP : CNNControllerPatch
#         # ------------------------------------------------------
#         elif is_cnn_patch:
#             if verbose:
#                 print(f"CNN mode: génération de {T - 1} pas de temps...")

#             for t in range(T - 1):
#                 if verbose and t % 10 == 0:
#                     print(f"  step {t}/{T - 1}")

#                 field_t = true_traj[t].unsqueeze(0)              # (1, N)
#                 patches = extract_spatial_patches(field_t, patch_radius)  # (1, N, P)
#                 patches_flat = patches.reshape(N, -1)            # (N, P)

#                 nu_vals = torch.full((N, 1), nu_value, device=device)
#                 pred_next = model(patches_flat, nu_vals)         # (N,)
#                 preds.append(pred_next)

#             pred_future = torch.stack(preds, dim=0)              # (T-1, N)
#             pred_traj = torch.cat([true_traj[0:1, :], pred_future], dim=0)

#         # ------------------------------------------------------
#         # 3) Autre type de modèle (par sécurité)
#         # ------------------------------------------------------
#         else:
#             raise TypeError(
#                 f"generate_model_predictions ne sait pas gérer le type de modèle {model_name} "
#                 f"(attendus : CNNControllerPatch, RNNControllerPatch, "
#                 f"TransformerController, CNNSpaceTimeController)."
#             )

#     if verbose:
#         print(f"\nGenerated predictions shape: {pred_traj.shape}")

#     return true_traj, pred_traj, nu_value

def evaluate_model_on_sample(model, train_loader, device, patch_radius, max_val=1.0, val_range=1.0, chunk_size=3, nu_target=None):
    """
    Evaluates a model on a sample and returns performance metrics.
    Compatible with CNN and RNN.
    
    Args:
        model: The trained CNN or RNN model
        train_loader: DataLoader containing the data
        device: PyTorch device (cuda or cpu) 
        patch_radius: Radius of spatial patches
        max_val: Maximum value for PSNR calculation
        val_range: Value range for SSIM calculation
            chunk_size: Size of temporal chunks for RNN (ignored for CNN)
            nu_target: If provided, tries to pick a sample with this viscosity value from the loader.
    
    Returns:
        dict: Dictionary containing metrics and trajectories
            - 'true_traj': Real trajectory
            - 'pred_traj': Predicted trajectory
            - 'nu_value': Viscosity value
            - 'psnr': PSNR score
            - 'ssim': SSIM score
            - 'mse': Mean squared error
            - 'r2': R² score
    """
    true_traj, pred_traj, nu_value = generate_model_predictions(
        model,
        train_loader,
        device,
        patch_radius,
        verbose=False,
        chunk_size=chunk_size,
        nu_target=nu_target,
    )
    
    # Calculate metrics
    psnr_score = psnr(true_traj, pred_traj, max_val=max_val)
    ssim_score = ssim(true_traj, pred_traj, val_range=val_range)
    mse_score = torch.mean((true_traj - pred_traj)**2).item()
    r2_score_val = r2_score(pred_traj, true_traj)
    
    return {
        'true_traj': true_traj,
        'pred_traj': pred_traj,
        'nu_value': nu_value,
        'psnr': psnr_score,
        'ssim': ssim_score,
        'mse': mse_score,
        'r2': r2_score_val
    }


def display_evaluation_results(evaluation_results, show_plots=True):
    """
    Displays evaluation metrics and error visualizations.
    
    Args:
        evaluation_results: Dictionary returned by evaluate_model_on_sample()
        show_plots: If True, displays error plots
        
    Returns:
        tuple: (true_traj, pred_traj) for later use if needed
    """
    true_traj = evaluation_results['true_traj']
    pred_traj = evaluation_results['pred_traj']
    
    print(f"Evaluation metrics:")
    print(f"  - PSNR: {evaluation_results['psnr']:.3f} dB")
    print(f"  - SSIM: {evaluation_results['ssim']:.3f}")
    print(f"  - MSE: {evaluation_results['mse']:.6f}")
    print(f"  - R²: {evaluation_results['r2']:.4f}")
    
    if show_plots:
        # Error visualization
        error = torch.abs(true_traj - pred_traj)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(error.cpu().numpy(), aspect='auto', cmap='hot')
        plt.colorbar(label='Absolute error')
        plt.xlabel('Spatial position')
        plt.ylabel('Time')
        plt.title('Absolute error map')
        
        plt.subplot(1, 2, 2)
        mean_error_per_time = torch.mean(error, dim=1).cpu().numpy()
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mean error')
        plt.title('Error evolution over time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return true_traj, pred_traj




def generate_model_predictions(model, train_loader, device, patch_radius,
                               verbose: bool = True, chunk_size: int = 3, nu_target=None, list_nus: bool = False, fallback_to_first: bool = True):
    """
    Version "confort" pour la visu : prend le premier sample du loader,
    appelle rollout_trajectory, et renvoie true_traj, pred_traj, nu.
    """
    # Small debugging helpers
    if list_nus:
        try:
            nus = get_loader_nu_values(train_loader)
            print(f"Loader contains {len(nus)} unique nu values: {nus}")
        except Exception as e:
            if verbose:
                print(f"Unable to enumerate loader nu values: {e}")

    # Pick a sample to visualize
    if nu_target is None:
        init_field, true_traj_batch, nu_batch = next(iter(train_loader))
    else:
        # try to find a batch where the first sample has the requested nu
        found = False
        for init_field, true_traj_batch, nu_batch in train_loader:
            try:
                this_nu = float(nu_batch[0].item())
            except Exception:
                this_nu = float(nu_batch)
            if abs(this_nu - float(nu_target)) < 1e-8:
                found = True
                break
        if not found:
            if fallback_to_first:
                if verbose:
                    print(f"No sample with nu={nu_target} found in the loader; falling back to the first sample.")
                init_field, true_traj_batch, nu_batch = next(iter(train_loader))
            else:
                raise ValueError(f"No sample with nu={nu_target} found in the loader")

    # Convert the picked batch into single trajectory and nu_value
    true_traj = true_traj_batch[0].to(device)   # (T, N)
    try:
        nu_value = float(nu_batch[0].item())
    except Exception:
        nu_value = float(nu_batch)

    if verbose:
        print(f"Generating predictions for nu = {nu_value:.4f}")
        print(f"True trajectory shape: {true_traj.shape}")
        print(f"Model type: {type(model).__name__}")

    pred_traj = rollout_trajectory(
        model=model,
        true_traj=true_traj,
        nu_value=nu_value,
        device=device,
        patch_radius=patch_radius,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    return true_traj, pred_traj, nu_value

def rollout_trajectory(
    model,
    true_traj: torch.Tensor,
    nu_value: float,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    verbose: bool = False,
):
    """
    Reconstruit une trajectoire prédite à partir d'une trajectoire vraie.

    true_traj : (T, N) sur device CPU ou GPU
    nu_value  : scalaire (float)
    Retour:
      pred_traj : (T, N)
    """
    from training import build_patches_from_sequence, extract_spatial_patches

    true_traj = true_traj.to(device)
    T, N = true_traj.shape
    patch_size = 2 * patch_radius + 1

    model.eval()
    model_name = type(model).__name__

    # CNN 1-step : patch spatial (B, P) + nu
    is_cnn_patch = model_name in ["CNNControllerPatch", "CNNController"]
    # Tout le reste = modèles séquentiels (RNN, CNNHistory, SingleChannelSpaceTimeCNN, CNNSpaceTimeController, etc.)
    is_seq_model = not is_cnn_patch

    preds = []

    with torch.no_grad():
        if is_seq_model:
            if T <= chunk_size:
                if verbose:
                    print(f"T={T} <= chunk_size={chunk_size}, on renvoie la vérité terrain.")
                pred_traj = true_traj.clone()
            else:
                for t in range(T - chunk_size):
                    if verbose and t % 10 == 0:
                        print(f"  step {t}/{T - chunk_size}")

                    # (1, L, N)
                    current_chunk = true_traj[t:t+chunk_size, :].unsqueeze(0)

                    # (N, L, P)
                    patches = build_patches_from_sequence(
                        current_chunk, patch_radius, patch_size
                    )

                    nu_vals = torch.full(
                        (patches.size(0), 1),
                        nu_value,
                        device=device,
                    )

                    pred_next = model(patches, nu_vals)  # (N,)
                    preds.append(pred_next)

                pred_future = torch.stack(preds, dim=0)         # (T - chunk_size, N)
                pred_traj = torch.cat(
                    [true_traj[:chunk_size, :], pred_future],
                    dim=0
                )                                              # (T, N)

        else:
            # --- CNN 1-step (type CNNControllerPatch)
            for t in range(T - 1):
                if verbose and t % 10 == 0:
                    print(f"  step {t}/{T - 1}")

                field_t = true_traj[t].unsqueeze(0)           # (1, N)
                patches = extract_spatial_patches(field_t, patch_radius)  # (1, N, P)
                patches_flat = patches.reshape(N, -1)         # (N, P)

                nu_vals = torch.full((N, 1), nu_value, device=device)
                pred_next = model(patches_flat, nu_vals)      # (N,)
                preds.append(pred_next)

            pred_future = torch.stack(preds, dim=0)           # (T - 1, N)
            pred_traj = torch.cat([true_traj[0:1, :], pred_future], dim=0)

    return pred_traj


def error_evolution_on_loader(
    model,
    data_loader,
    device,
    patch_radius: int,
    chunk_size: int = 3,
    show_plot: bool = True,
):
    """
    Calcule l'évolution de l'erreur moyenne dans le temps, moyennée sur tout le data_loader.

    Retourne:
      mean_error_per_time : numpy array de shape (T,)
    """
    model.eval()

    sum_error_per_time = None   # accumulation sur tous les samples
    n_samples = 0
    T_ref = None

    for init_field, traj_batch, nu_batch in data_loader:
        B, T, N = traj_batch.shape

        if T_ref is None:
            T_ref = T

        for b in range(B):
            true_traj = traj_batch[b].to(device)        # (T, N)
            nu_value = float(nu_batch[b].item())

            # Trajectoire prédite
            pred_traj = rollout_trajectory(
                model=model,
                true_traj=true_traj,
                nu_value=nu_value,
                device=device,
                patch_radius=patch_radius,
                chunk_size=chunk_size,
                verbose=False,
            )

            # On tronque si jamais T varie (par sécurité)
            T_eff = min(true_traj.shape[0], pred_traj.shape[0], T_ref)
            true_traj_eff = true_traj[:T_eff, :]
            pred_traj_eff = pred_traj[:T_eff, :]

            # Erreur absolue : (T_eff, N)
            error = torch.abs(true_traj_eff - pred_traj_eff)

            # Moyenne spatiale -> (T_eff,)
            error_per_time = torch.mean(error, dim=1)   # (T_eff,)

            if sum_error_per_time is None:
                sum_error_per_time = torch.zeros_like(error_per_time)

            sum_error_per_time[:T_eff] += error_per_time
            n_samples += 1

    if n_samples == 0:
        raise RuntimeError("error_evolution_on_loader: no samples in data_loader")

    mean_error_per_time = (sum_error_per_time / n_samples).cpu().numpy()

    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mean absolute error')
        plt.title('Error evolution over time (averaged over test set)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return mean_error_per_time
