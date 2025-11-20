# data_loading.py
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

# -------- Dataset "simple" = version notebooks cnn/rnn --------

class BurgersDatasetSimple(Dataset):
    """
    Dataset proche de cnn.py/rnn.py :
    - charge quelques fichiers .npz (un par viscosité)
    - renvoie (initial_field, full_trajectory, nu)
    full_trajectory : (T, N)
    initial_field   : (N,)
    nu              : (1,)
    """
    def __init__(self, files_by_nu: dict[float, str | Path]):
        self.trajectories = []
        self.nu_values = []

        for nu, path in files_by_nu.items():
            uu_tensor = np.load(str(Path(path).resolve()))["u"]  # (T, N)
            data_tensor = torch.tensor(uu_tensor, dtype=torch.float32)
            self.trajectories.append(data_tensor)   # une seule traj par nu ici
            self.nu_values.append(nu)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]   # (T, N)
        initial_field = traj[0]         # (N,)
        nu = torch.tensor([self.nu_values[idx]], dtype=torch.float32)
        return initial_field, traj, nu


# -------- Dataset "généré" = version sam_cnn --------

class BurgersViscosityDataset(Dataset):
    """
    Regroupe beaucoup de trajectoires par viscosité, comme dans sam_cnn.py.
    Renvoie (initial_field, full_trajectory, nu).
    """
    def __init__(self, datasets: list[torch.Tensor], viscosities: list[float]):
        # datasets: list de tensors (num_samples, T, N)
        self.data = []
        self.nu = []
        for data, nu in zip(datasets, viscosities):
            num_samples = data.shape[0]
            self.data.append(data)
            self.nu.append(torch.full((num_samples, 1), nu, dtype=torch.float32))

        self.data = torch.cat(self.data, dim=0)  # (total_samples, T, N)
        self.nu = torch.cat(self.nu, dim=0)      # (total_samples, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        full_trajectory = self.data[idx]        # (T, N)
        initial_field = full_trajectory[0, :]   # (N,)
        nu = self.nu[idx]                       # (1,)
        return initial_field, full_trajectory, nu


# --------- Helpers pour charger les .npz "sam_cnn" ---------

def _extract_U_and_nu_from_npz(path: Path):
    data = np.load(str(path), allow_pickle=True)
    if "nu" in data.files:
        nu = float(data["nu"])
    else:
        nu = 0.1
    if "U" in data.files:
        U = np.asarray(data["U"])
    elif "u" in data.files:
        U = np.asarray(data["u"])
    else:
        # fallback
        for k in data.files:
            if k not in ("x", "t", "dx", "dt", "nu", "speed", "tag"):
                U = np.asarray(data[k])
                break
        else:
            U = np.asarray(data[data.files[0]])

    if U.ndim == 1:
        U = U[None, :]
    elif U.ndim > 2:
        U = U.reshape(U.shape[0], -1)

    return U, nu


def collect_generated_burgers(root_dir: str, history_len: int):
    """
    Version simplifiée de collect_generated_burgers() de sam_cnn.py.
    - Filtre les fichiers trop courts (T < history_len+1)
    - Groupe par viscosité
    - Aligne le temps par troncature et l'espace par padding/crop.
    """
    root = Path(root_dir)
    files = sorted(root.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun .npz trouvé dans {root}")
    MIN_T = history_len + 1

    groups = {}
    shapes = {}
    for p in files:
        U, nu = _extract_U_and_nu_from_npz(p)
        T, N = U.shape
        if T < MIN_T:
            continue
        groups.setdefault(nu, []).append(U)
        shapes.setdefault(nu, []).append((T, N))

    datasets = []
    viscosities = []
    for nu, arrs in groups.items():
        Ts = [a.shape[0] for a in arrs]
        Ns = [a.shape[1] for a in arrs]
        target_T = min(Ts)
        target_N = max(Ns)
        aligned = []
        for a in arrs:
            a2 = _align_array_to_shape(a, target_T, target_N)
            aligned.append(a2)
        arr_tensor = torch.tensor(np.stack(aligned, axis=0), dtype=torch.float32)
        datasets.append(arr_tensor)
        viscosities.append(float(nu))
    return datasets, viscosities


def _align_array_to_shape(arr, target_T, target_N):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    T, N = arr.shape
    # temps : tronque si trop long
    if T > target_T:
        arr = arr[:target_T, :]
    elif T < target_T:
        raise ValueError("T trop petit pour l'alignement")

    # espace : pad symétrique ou crop
    if N < target_N:
        pad = target_N - N
        left = pad // 2
        right = pad - left
        arr = np.pad(arr, ((0, 0), (left, right)), mode="edge")
    elif N > target_N:
        excess = N - target_N
        left = excess // 2
        right = left + target_N
        arr = arr[:, left:right]
    return arr


# --------- Fabriques de DataLoader ---------

def create_simple_dataloader(
    files_by_nu: dict[float, str | Path],
    batch_size: int,
    shuffle: bool = True,
):
    dataset = BurgersDatasetSimple(files_by_nu)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def create_generated_dataloaders(
    root_dir: str,
    history_len: int,
    batch_size: int,
    train_ratio: float = 0.8,
):
    datasets, viscosities = collect_generated_burgers(root_dir, history_len)
    full_dataset = BurgersViscosityDataset(datasets, viscosities)
    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# --------- Fonction pour regarder rapidement le dataset ---------

def show_random_sample(dataset, idx: int | None = None):
    """
    Affiche la shape d'un exemple + un petit heatmap de la trajectoire.
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    initial_field, traj, nu = dataset[idx]
    print(f"Sample {idx} | nu={float(nu):.4f}")
    print(" initial_field:", initial_field.shape)
    print(" trajectory   :", traj.shape)

    traj_np = traj.numpy().T  # (space, time)
    plt.figure(figsize=(6, 3))
    plt.imshow(traj_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("time")
    plt.ylabel("space")
    plt.title(f"Trajectory (nu={float(nu):.4f})")
    plt.tight_layout()
    plt.show()
