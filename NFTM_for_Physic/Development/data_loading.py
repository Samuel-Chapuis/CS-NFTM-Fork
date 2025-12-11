# data_loading.py
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

# -------- BurgersViscosityDataset --------
# Alex use
class BurgersDataset(Dataset):
    def __init__(self, files_burger):
        # files_burger: dict {nu: path}
        self.trajs = [] # list of (T,N)
        self.nu_values = []   # scalar viscosities, one per traj

        for nu, path in files_burger.items():
            data = np.load(str(path.resolve()))['u'] # (100, T, N)
            data_t = torch.tensor(data, dtype = torch.float32) # (100, T, N)
    
            for traj in data_t: # traj: (T,N)
                self.trajs.append(traj)
                self.nu_values.append(nu)
        
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        # Returns: tuple (initial_field (N,), full_trajectory (T, N), viscosity (1,))
        traj = self.trajs[idx]  # (T, N)
        init_field = traj[0]  # (N,)
        nu_val = torch.tensor([self.nu_values[idx]], dtype=torch.float32)
        return init_field, traj, nu_val

# Samuel use
class BurgersViscosityDataset():
    """
    Groups many trajectories by viscosity, like in sam_cnn.py.
    Returns (initial_field, full_trajectory, nu).
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


# --------- Helpers to load the .npz "sam_cnn" files ---------

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
    Simplified version of collect_generated_burgers() from sam_cnn.py.
    - Filters files that are too short (T < history_len+1)
    - Groups by viscosity
    - Aligns time by truncation and space by padding/crop.
    """
    root = Path(root_dir)
    files = sorted(root.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root}")
    MIN_T = history_len + 1

    groups = {}
    shapes = {}
    for p in files:
        try:
            U, nu = _extract_U_and_nu_from_npz(p)
        except Exception as e:
            print(f"❌ Skipping file (invalid or corrupted .npz): {p} | error: {e}")
            continue
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


def validate_npz_files(root_dir: str, max_examples: int | None = 20):
    """
    Scans .npz files under root_dir and returns two lists: good_files and bad_files.
    Each item in good_files contains path, size (bytes), keys, and nu (if present).
    Each item in bad_files contains path and error string.

    This is useful to quickly locate corrupted or incompatible npz files.
    """
    from pathlib import Path
    root = Path(root_dir)
    files = sorted(root.glob("**/*.npz"))
    good = []
    bad = []
    for i, p in enumerate(files):
        try:
            size = p.stat().st_size
            data = np.load(str(p), allow_pickle=True)
            keys = list(data.files)
            nu = float(data["nu"]) if "nu" in data.files else None
            good.append({"path": str(p), "size": size, "keys": keys, "nu": nu})
        except Exception as e:
            bad.append({"path": str(p), "error": str(e)})
        if max_examples is not None and i >= max_examples:
            break
    return good, bad


def _align_array_to_shape(arr, target_T, target_N):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    T, N = arr.shape
    # time: truncate if too long
    if T > target_T:
        arr = arr[:target_T, :]
    elif T < target_T:
        raise ValueError("T too small for alignment")

    # space: symmetric pad or crop
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




def create_generated_dataloaders(
    root_dir: str,
    history_len: int,
    batch_size: int,
    train_ratio: float = 0.8,
):
    """
    DEPRECATED: Use create_generated_dataloaders_from_folders for separate train/test folders.
    Creates train/test loaders from a single directory with random split.
    """
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


def create_generated_dataloaders_from_folders(
    train_dir: str,
    test_dir: str,
    history_len: int,
    batch_size: int,
):
    """
    Creates train/test loaders from separate train and test directories.
    
    Args:
        train_dir: Path to the training data folder
        test_dir: Path to the test data folder
        history_len: Minimum history length required
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, test_loader
    """
    # Load training data
    train_datasets, train_viscosities = collect_generated_burgers(train_dir, history_len)
    train_dataset = BurgersViscosityDataset(train_datasets, train_viscosities)
    
    # Load test data
    test_datasets, test_viscosities = collect_generated_burgers(test_dir, history_len)
    test_dataset = BurgersViscosityDataset(test_datasets, test_viscosities)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# --------- Function to quickly look at the dataset ---------

def show_random_sample(dataset, idx: int):
    """
    Displays the shape of an example + a small heatmap of the trajectory.
    """
    # if idx is None:
    #     idx = random.randint(0, len(dataset) - 1)
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
