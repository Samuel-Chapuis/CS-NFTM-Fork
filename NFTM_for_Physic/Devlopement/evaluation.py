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


def generate_model_predictions(model, train_loader, device, patch_radius, verbose=True, chunk_size=3):
    """
    Génère les prédictions d'un modèle pour un échantillon du train_loader.
    Compatible avec CNN et RNN.
    
    Args:
        model: Le modèle CNN ou RNN entraîné
        train_loader: DataLoader contenant les données
        device: Device PyTorch (cuda ou cpu)
        patch_radius: Rayon des patches spatiaux
        verbose: Affichage des informations de progression
        chunk_size: Taille des chunks temporels pour RNN (ignoré pour CNN)
    
    Returns:
        tuple: (true_traj, pred_traj, nu_value)
            - true_traj: Trajectoire réelle (T, N)
            - pred_traj: Trajectoire prédite (T, N)
            - nu_value: Valeur de viscosité utilisée
    """
    from models import CNNControllerPatch, RNNControllerPatch
    from training import build_patches_from_sequence
    
    # Préparation des données pour la prédiction
    init_field, true_traj, nu = next(iter(train_loader))
    true_traj = true_traj[0].to(device)  # Prendre le premier échantillon du batch
    nu_value = float(nu[0])

    if verbose:
        print(f"Génération de prédictions pour nu = {nu_value:.4f}")
        print(f"Shape de la trajectoire vraie: {true_traj.shape}")
        print(f"Type de modèle: {type(model).__name__}")

    # Génération des prédictions
    model.eval()
    with torch.no_grad():
        T, N = true_traj.shape
        preds = []
        
        # Détection du type de modèle
        is_rnn = isinstance(model, RNNControllerPatch)
        patch_size = 2 * patch_radius + 1
        
        if is_rnn:
            # Pour RNN: utiliser des chunks temporels
            if verbose:
                print(f"Mode RNN: Génération de {T-chunk_size} prédictions avec chunk_size={chunk_size}...")
            
            for t in range(T - chunk_size):
                if verbose and t % 10 == 0:
                    print(f"  Pas {t}/{T-chunk_size}")
                
                # Chunk temporel: (chunk_size, N)
                current_chunk = true_traj[t:t+chunk_size, :].unsqueeze(0)  # (1, chunk_size, N)
                
                # Extraction des patches pour chaque point spatial
                # patches: (1*N, chunk_size, patch_size)
                patches = build_patches_from_sequence(current_chunk, patch_radius, patch_size)
                
                # Nu values pour tous les points spatiaux
                nu_vals = torch.full((N, 1), nu_value, device=device)  # (N, 1)
                
                # Prédiction
                pred_next = model(patches, nu_vals)  # (N,)
                preds.append(pred_next)
                
            # Reconstruction de la trajectoire
            if preds:
                pred_traj = torch.stack(preds, dim=0)  # (T-chunk_size, N)
                # Ajouter les premiers pas de temps (ground truth)
                pred_traj = torch.cat([true_traj[:chunk_size, :], pred_traj], dim=0)  # (T, N)
            else:
                pred_traj = true_traj.clone()  # Fallback si pas assez de données
                
        else:
            # Pour CNN: prédiction pas à pas
            if verbose:
                print(f"Mode CNN: Génération de {T-1} pas de temps...")
            
            for t in range(T - 1):
                if verbose and t % 10 == 0:
                    print(f"  Pas {t}/{T-1}")
                    
                field_t = true_traj[t].unsqueeze(0)  # (1, N)
                patches = extract_spatial_patches(field_t, patch_radius)      # (1, N, P)
                patches_flat = patches.reshape(N, -1)                         # (N, P)
                nu_vals = torch.full((N, 1), nu_value, device=device)         # (N, 1)
                pred_next = model(patches_flat, nu_vals)                      # (N,)
                preds.append(pred_next)
                
            pred_traj = torch.stack(preds, dim=0)                             # (T-1, N)
            pred_traj = torch.cat([true_traj[0:1, :], pred_traj], dim=0)      # (T, N)

    if verbose:
        print(f"\nPrédictions générées: {pred_traj.shape}")
    
    return true_traj, pred_traj, nu_value


def evaluate_model_on_sample(model, train_loader, device, patch_radius, max_val=1.0, val_range=1.0, chunk_size=3):
    """
    Évalue un modèle sur un échantillon et retourne les métriques de performance.
    Compatible avec CNN et RNN.
    
    Args:
        model: Le modèle CNN ou RNN entraîné
        train_loader: DataLoader contenant les données
        device: Device PyTorch (cuda ou cpu) 
        patch_radius: Rayon des patches spatiaux
        max_val: Valeur maximale pour le calcul du PSNR
        val_range: Plage de valeurs pour le calcul du SSIM
        chunk_size: Taille des chunks temporels pour RNN (ignoré pour CNN)
    
    Returns:
        dict: Dictionnaire contenant les métriques et les trajectoires
            - 'true_traj': Trajectoire réelle
            - 'pred_traj': Trajectoire prédite
            - 'nu_value': Valeur de viscosité
            - 'psnr': Score PSNR
            - 'ssim': Score SSIM
            - 'mse': Erreur quadratique moyenne
            - 'r2': Score R²
    """
    true_traj, pred_traj, nu_value = generate_model_predictions(
        model, train_loader, device, patch_radius, verbose=False, chunk_size=chunk_size
    )
    
    # Calcul des métriques
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
    Affiche les métriques d'évaluation et les visualisations d'erreur.
    
    Args:
        evaluation_results: Dictionnaire retourné par evaluate_model_on_sample()
        show_plots: Si True, affiche les graphiques d'erreur
        
    Returns:
        tuple: (true_traj, pred_traj) pour usage ultérieur si nécessaire
    """
    true_traj = evaluation_results['true_traj']
    pred_traj = evaluation_results['pred_traj']
    
    print(f"Métriques d'évaluation:")
    print(f"  - PSNR: {evaluation_results['psnr']:.3f} dB")
    print(f"  - SSIM: {evaluation_results['ssim']:.3f}")
    print(f"  - MSE: {evaluation_results['mse']:.6f}")
    print(f"  - R²: {evaluation_results['r2']:.4f}")
    
    if show_plots:
        # Visualisation des erreurs
        error = torch.abs(true_traj - pred_traj)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(error.cpu().numpy(), aspect='auto', cmap='hot')
        plt.colorbar(label='Erreur absolue')
        plt.xlabel('Position spatiale')
        plt.ylabel('Temps')
        plt.title('Carte d\'erreur absolue')
        
        plt.subplot(1, 2, 2)
        mean_error_per_time = torch.mean(error, dim=1).cpu().numpy()
        plt.plot(mean_error_per_time, 'r-', linewidth=2)
        plt.xlabel('Temps')
        plt.ylabel('Erreur moyenne')
        plt.title('Évolution de l\'erreur dans le temps')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return true_traj, pred_traj
