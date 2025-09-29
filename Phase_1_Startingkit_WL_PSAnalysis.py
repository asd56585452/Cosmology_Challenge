# %% [markdown]
# # FAIR Universe - Weak Lensing ML Uncertainty Challenge
# ## Phase 1 Starting Kit: Power Spectrum Analysis
# ***
# 
# In the universe, massive objects like galaxies and clusters of galaxies bend the path of light traveling near them — a phenomenon known as gravitational lensing, as predicted by Einstein’s general relativity. When this bending is subtle, it’s known as weak gravitational lensing. It slightly distorts the shapes of distant galaxies as their light travels through the cosmic web of matter, including dark matter and ordinary matter (baryons), on its way to Earth.
# 
# By carefully measuring these tiny distortions across the sky, we can reconstruct convergence maps — essentially cosmic “heat maps” that show where matter is concentrated, even if that matter is invisible. These maps let us trace the hidden structure of the universe.
# 
# Crucially, weak lensing maps contain rich cosmological information. By statistically analyzing the patterns in these distortions, we can learn about the universe’s content and evolution. For example, we can estimate how much dark matter and dark energy exist, how fast the universe is expanding, and how structures have grown over time. In this way, weak lensing helps us constrain our cosmological model.
# 
# Currently, the most widely accepted model of the universe is called $\Lambda$CDM (Lambda Cold Dark Matter). It describes a universe dominated by dark energy (represented by the Greek letter $\Lambda$, or Lambda) and cold dark matter, with only a small fraction made up of normal matter like stars and planets. Weak lensing is one of the most powerful tools we have to test and refine this model.
# 
# The goal of this data challenge is to weak lensing convergence maps to constrain the physical parameters of ΛCDM model, $\Omega_m$, which describes what fraction of the universe’s total energy is made of matter (both normal and dark matter), and $S_8$, which measures of how “clumpy” the matter in the universe is on large scales.
# 
# ***

# %% [markdown]
# `COLAB` determines whether this notebook is running on Google Colab.

# %%
# COLAB = 'google.colab' in str(get_ipython())

# %%
# if COLAB:
#     # clone github repo
#     !git clone --depth 1 https://github.com/FAIR-Universe/Cosmology_Challenge.git
#     # move to the HEP starting kit folder
#     %cd Cosmology_Challenge/

# %% [markdown]
# # 0 - Imports & Settings

# %%
import os
import json
import time
import zipfile
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import optuna
# %matplotlib inline

# %% [markdown]
# # 1 - Helper Classes for
# - Utitlity Functions
# - Data Loading
# - Visualizations
# - Scoring Functions

# %% [markdown]
# ### Utility

# %%
class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        """
        Add noise to a noiseless convergence map.

        Parameters
        ----------
        data : np.array
            Noiseless convergence maps.
        mask : np.array
            Binary mask map.
        ng : float
            Number of galaxies per arcmin². This determines the noise level; a larger number means smaller noise.
        pixel_size : float, optional
            Pixel size in arcminutes (default is 2.0).
        """

        return data + np.random.randn(*data.shape) * 0.4 / (2*ng*pixel_size**2)**0.5 * mask

    @staticmethod
    def load_np(data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_np(data_dir, file_name, data):
        file_path = os.path.join(data_dir, file_name)
        np.save(file_path, data)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        """
        Save a dictionary with 'means' and 'errorbars' into a JSON file,
        then compress it into a ZIP file inside submission_dir.

        Parameters
        ----------
        submission_dir : str
            Path to the directory where the ZIP file will be saved.
        file_name : str
            Name of the ZIP file (without extension).
        data : dict
            Dictionary with keys 'means' and 'errorbars'.

        Returns
        -------
        str
            Path to the created ZIP file.
        """
        os.makedirs(submission_dir, exist_ok=True)

        json_path = os.path.join(submission_dir, json_file_name)

        # Save JSON file
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Path to ZIP
        zip_path = os.path.join(submission_dir, zip_file_name)

        # Create ZIP containing only the JSON
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)

        # Remove the standalone JSON after zipping
        os.remove(json_path)

        return zip_path

# %% [markdown]
# ### Data

# %%
class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        self.viz_label_file = 'label.npy'
        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = self.viz_label_file
            self.Ncosmo = 101  # Number of cosmologies in the entire training data
            self.Nsys = 256    # Number of systematic realizations in the entire training data
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000  # Number of instances in the test data
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3    # Number of cosmologies in the sampled training data
            self.Nsys = 20     # Number of systematic realizations in the sampled training data
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3     # Number of instances in the sampled test data

        self.shape = [1424,176] # dimensions of each map
        self.pixelsize_arcmin = 2 # pixel size in arcmin
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi # pixel size in radian
        self.ng = 30  # galaxy number density. This determines the noise level of the experiment. Do not change this number.

    def load_train_data(self):
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file) # A binary map that shows which parts of the sky are observed and which areas are blocked
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:,:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.kappa_file) # Training convergence maps
        self.label = Utility.load_np(data_dir=self.data_dir, file_name=self.label_file) # Training labels (cosmological and physical paramameters) of each training map
        self.viz_label = Utility.load_np(data_dir=self.data_dir, file_name=self.viz_label_file) # For visualization of parameter distributions

    def load_test_data(self):
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.test_kappa_file) # Test noisy convergence maps

# %% [markdown]
# ### Scoring function

# %%
class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        """
        Computes the log-likelihood score for Phase 1 based on predicted cosmological parameters.

        Parameters
        ----------
        true_cosmo : np.ndarray
            Array of true cosmological parameters (shape: [n_samples, n_params]).
        infer_cosmo : np.ndarray
            Array of inferred cosmological parameters from the model (same shape as true_cosmo).
        errorbar : np.ndarray
            Array of standard deviations (uncertainties) for each inferred parameter
            (same shape as true_cosmo).

        Returns
        -------
        np.ndarray
            Array of scores for each sample (shape: [n_samples]).
        """

        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000  # This is a constant that scales the error term.
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        if score >= -10**6: # Set a minimum of the score (to properly display on Codabench)
            return score
        else:
            return -10**6

# %% [markdown]
# # 2 - Load train and test data

# %% [markdown]
# The training maps are generated by $N_{\rm cosmo}$ cosmological models, each model contains $N_{\rm sys}$ realizations with different nuisance parameters. So the shape of the training maps kappa is $(N_{\rm cosmo}, N_{\rm sys}, 1424, 176)$.
# 
# Each training map is associated with 5 parameters, so the shape of label is $(N_{\rm cosmo}, N_{\rm sys}, 5)$. The first two parameters are cosmological parameters $\Omega_m$ and $S_8$, while the rest three parameters are nuisance parameters that describe systematic effects and need to be marginalized in the data analysis (two of them describe baryonic effects and the last one describes photometric redshift uncertainties).

# %% [markdown]
# # 2 - Load train and test data

# %% [markdown]
# The training maps are generated by $N_{\rm cosmo}$ cosmological models, each model contains $N_{\rm sys}$ realizations with different nuisance parameters. So the shape of the training maps kappa is $(N_{\rm cosmo}, N_{\rm sys}, 1424, 176)$.
# 
# Each training map is associated with 5 parameters, so the shape of label is $(N_{\rm cosmo}, N_{\rm sys}, 5)$. The first two parameters are cosmological parameters $\Omega_m$ and $S_8$, while the rest three parameters are nuisance parameters that describe systematic effects and need to be marginalized in the data analysis (two of them describe baryonic effects and the last one describes photometric redshift uncertainties).

# %% [markdown]
# # 4 - Training (build power spectrum and covariance emulator)

# %% [markdown]
# In cosmology, the power spectrum describes how matter is distributed across different size scales in the universe and is a key tool for studying the growth of cosmic structure. Starting from the matter density $\delta(x)$, we transform it into Fourier space to get $\tilde{\delta}(x)$, which represents fluctuations as waves of different wavelengths. The matter power spectrum P(k) is then defined by:
# 
# \begin{equation}
# \langle \tilde{\delta}(\mathbf{k}) \tilde{\delta}^*(\mathbf{k}') \rangle = (2\pi)^3 \delta_D(\mathbf{k}-\mathbf{k}') P(k),
# \end{equation}
# 
# where k is the wavenumber corresponding to a scale $\lambda \sim 1/k$, and $ \delta_D$ is the Dirac delta function. Intuitively, P(k) tells us how "clumpy" the universe is on different scales. In cosmology, the shape and amplitude of P(k) encodes the physics and composition of the universe, making it one of the most important statistical tools in the field.
#
# In this notebook we use a CNN to constrain the cosmological parameters.

# %% [markdown]
# ### PyTorch Dataset Class

# %%
class WeakLensingDataset(Dataset):
    def __init__(self, kappa_path, label_path, sys_indices, data_obj, train=True):
        self.train = train
        self.mask = data_obj.mask
        self.ng = data_obj.ng
        self.pixelsize_arcmin = data_obj.pixelsize_arcmin
        self.shape = data_obj.shape
        self.Nsys_total = data_obj.Nsys # Total number of systematics in the full dataset

        # Store the list of system indices for this dataset (train or val)
        self.sys_indices = sys_indices

        # Open the complete, but flattened, numpy arrays using memory-mapping
        self.flat_maps = np.load(kappa_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')

        self.Ncosmo = self.labels.shape[0]
        self.Nsys_per_cosmo = len(self.sys_indices)
        self.unmasked_size = np.sum(self.mask)

    def __len__(self):
        return self.Ncosmo * self.Nsys_per_cosmo

    def __getitem__(self, idx):
        # Calculate 2D index from flat index
        cosmo_idx = idx // self.Nsys_per_cosmo
        list_idx = idx % self.Nsys_per_cosmo

        # Look up the original system index from our list
        original_sys_idx = self.sys_indices[list_idx]

        # Get the 1D slice of unmasked data for the specific map
        data_slice = self.flat_maps[cosmo_idx, original_sys_idx]

        # Reconstruct the 2D image on the fly
        map_data = np.zeros(self.shape, dtype=np.float64)
        map_data[self.mask] = data_slice

        label = self.labels[cosmo_idx, original_sys_idx, :2].astype(np.float32)

        # Noise addition is now handled on the GPU in the training loop

        # Convert to torch tensors and add channel dimension for CNN
        map_tensor = torch.from_numpy(map_data).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float()

        return map_tensor, label_tensor

# %% [markdown]
# ### CNN Model Definition

# %%
class DynamicCNN(nn.Module):
    def __init__(self, nf_scalings, layer_counts):
        super(DynamicCNN, self).__init__()
        nf = 16

        features = nn.ModuleList()
        in_c = 1

        for i in range(len(layer_counts)):
            out_c = int(nf * (2 ** i) * nf_scalings[i])
            if layer_counts[i] == 1:
                features.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
                ))
                in_c = out_c
            elif layer_counts[i] == 2:
                features.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=1), nn.BatchNorm2d(out_c//2), nn.ReLU()
                ))   
                in_c = out_c//2
            elif layer_counts[i] == 3:
                features.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=0), nn.BatchNorm2d(out_c//2), nn.ReLU(),
                nn.Conv2d(out_c//2, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
                ))
                in_c = out_c
            elif layer_counts[i] == 4:
                features.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=0), nn.BatchNorm2d(out_c//2), nn.ReLU(),
                nn.Conv2d(out_c//2, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=0), nn.BatchNorm2d(out_c//2), nn.ReLU()
                ))
                in_c = out_c//2
            elif layer_counts[i] == 5:
                features.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=0), nn.BatchNorm2d(out_c//2), nn.ReLU(),
                nn.Conv2d(out_c//2, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c//2, 1, padding=0), nn.BatchNorm2d(out_c//2), nn.ReLU(),
                nn.Conv2d(out_c//2, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
                ))
                in_c = out_c
            
            if i < len(layer_counts) - 1:
                features.append(nn.AvgPool2d(2, 2))


        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_c, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# %% [markdown]
# ### Loss Function

# %%
def gaussian_nll_loss(output, target):
    """
    Gaussian Negative Log-Likelihood loss.
    The output tensor is expected to have 4 columns:
    - 0: mean for Omega_m
    - 1: log_var for Omega_m
    - 2: mean for S_8
    - 3: log_var for S_8
    """
    # Separate means and log variances
    mean_om, log_var_om = output[:, 0], output[:, 1]
    mean_s8, log_var_s8 = output[:, 2], output[:, 3]

    # Get targets
    target_om, target_s8 = target[:, 0], target[:, 1]

    # Calculate variance
    var_om = torch.exp(log_var_om)
    var_s8 = torch.exp(log_var_s8)

    # Calculate loss for each parameter
    loss_om = 0.5 * (log_var_om + (target_om - mean_om)**2 / var_om)
    loss_s8 = 0.5 * (log_var_s8 + (target_s8 - mean_s8)**2 / var_s8)

    # Return the mean of the total loss
    return (loss_om + loss_s8).mean()

# %% [markdown]
# ### Prediction on Test Set

# %%
def add_noise_torch(data, mask, ng, pixel_size=2.):
    """
    Add noise to a noiseless convergence map tensor on the GPU.

    Parameters
    ----------
    data : torch.Tensor
        Noiseless convergence map tensor.
    mask : torch.Tensor
        Binary mask map tensor.
    ng : float
        Number of galaxies per arcmin².
    pixel_size : float, optional
        Pixel size in arcminutes (default is 2.0).
    """
    noise = torch.randn_like(data) * 0.4 / (2 * ng * pixel_size**2)**0.5
    return data + noise * mask

def predict(model, data_obj, device, batch_size):
    model.eval()
    all_test_preds = []

    # Create a simple dataloader for the test set
    # The test data is already noisy. We don't need the custom dataset class here.
    test_maps_tensor = torch.from_numpy(data_obj.kappa_test).float().unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_maps_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for maps in test_loader:
            maps = maps[0].to(device)
            outputs = model(maps)
            all_test_preds.append(outputs.cpu().numpy())

    all_test_preds = np.concatenate(all_test_preds, axis=0)

    # Extract mean and errorbar from predictions
    mean = all_test_preds[:, [0, 2]]
    log_var = all_test_preds[:, [1, 3]]
    errorbar = np.sqrt(np.exp(log_var))

    return mean, errorbar

def objective(trial, data_obj, device, mask_tensor, train_indices, val_indices, kappa_path, label_path, n_epochs):
    # -- Hyperparameters to Tune --
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 4 ,32 , log=True)
    nf_scalings = [trial.suggest_float(f"block_{i}_nf_scaling", 0.25, 4, log=True) for i in range(6)]
    layer_counts = [trial.suggest_int(f"block_{i}_layers", 1, 5) for i in range(6)]

    # -- Create Datasets and DataLoaders --
    train_dataset = WeakLensingDataset(
        kappa_path=kappa_path, label_path=label_path,
        sys_indices=train_indices, data_obj=data_obj, train=True
    )
    val_dataset = WeakLensingDataset(
        kappa_path=kappa_path, label_path=label_path,
        sys_indices=val_indices, data_obj=data_obj, train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # -- Model, Optimizer --
    model = DynamicCNN(nf_scalings=nf_scalings, layer_counts=layer_counts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_score = -np.inf  # Initialize with a very low value

    # -- Training Loop --
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{n_epochs} [Train]")
        for maps, labels in train_iterator:
            maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            optimizer.zero_grad()
            outputs = model(maps)
            loss = gaussian_nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * maps.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        val_iterator = tqdm(val_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{n_epochs} [Val]")
        with torch.no_grad():
            for maps, labels in val_iterator:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                outputs = model(maps)
                loss = gaussian_nll_loss(outputs, labels)
                val_loss += loss.item() * maps.size(0)
                all_val_preds.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)
        pred_mean = all_val_preds[:, [0, 2]]
        pred_log_var = all_val_preds[:, [1, 3]]
        pred_errorbar = np.sqrt(np.exp(pred_log_var))
        val_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)

        if val_score > best_val_score:
            best_val_score = val_score

        print(f"Trial {trial.number} Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}, Best Val Score: {best_val_score:.4f}")

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_score

def main():
    root_dir = os.getcwd()
    print("Root directory is", root_dir)
    USE_PUBLIC_DATASET = True
    PUBLIC_DATA_DIR = 'public_data/'
    DATA_DIR = PUBLIC_DATA_DIR if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')
    N_EPOCHS = 10
    N_TRIALS = 2
    N_JOBS = 2
    TIMEOUT = 600  # 1 hour

    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.mask = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.mask_file)
    if not USE_PUBLIC_DATASET: # viz_label is only available for the public dataset
        pass
    else:
        data_obj.viz_label = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.viz_label_file)
    data_obj.load_test_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    Nsys = data_obj.Nsys
    indices = np.arange(Nsys)
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    kappa_path = os.path.join(DATA_DIR, data_obj.kappa_file)
    label_path = os.path.join(DATA_DIR, data_obj.label_file)

    study = optuna.create_study(
        study_name="weak_lensing_phase1",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda trial: objective(trial, data_obj, device, mask_tensor, train_indices, val_indices, kappa_path, label_path, N_EPOCHS),n_jobs=N_JOBS, n_trials=N_TRIALS, timeout=TIMEOUT)

    print("Best trial:", study.best_trial.params)

    # Save the best hyperparameters
    best_params = study.best_trial.params
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best hyperparameters saved to best_hyperparameters.json")

    # Train final model with best params
    model = DynamicCNN(nf_scalings=[best_params[f'block_{i}_nf_scaling'] for i in range(6)], layer_counts=[best_params[f'block_{i}_layers'] for i in range(6)]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    # Create a new train/val split for the final training
    final_train_indices, final_val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    final_train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=final_train_indices, data_obj=data_obj, train=True)
    final_val_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=final_val_indices, data_obj=data_obj, train=True)
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
    final_val_loader = DataLoader(final_val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=0, pin_memory=False)

    best_val_score = -np.inf
    best_model_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        train_iterator = tqdm(final_train_loader, desc=f"Final Training Epoch {epoch+1}/{N_EPOCHS}")
        for maps, labels in train_iterator:
            maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            optimizer.zero_grad()
            outputs = model(maps)
            loss = gaussian_nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * maps.size(0)
        train_loss /= len(final_train_loader.dataset)

        # Validation for final model
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for maps, labels in final_val_loader:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                outputs = model(maps)
                loss = gaussian_nll_loss(outputs, labels)
                val_loss += loss.item() * maps.size(0)
                all_val_preds.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
        val_loss /= len(final_val_loader.dataset)

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)
        pred_mean = all_val_preds[:, [0, 2]]
        pred_log_var = all_val_preds[:, [1, 3]]
        pred_errorbar = np.sqrt(np.exp(pred_log_var))
        val_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)

        print(f"Final Training Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')
            print(f"New best model saved with score: {best_val_score:.4f}")


    print("Loading best model for prediction...")
    model.load_state_dict(torch.load('best_model.pth'))

    print("Generating predictions on the test set...")
    mean, errorbar = predict(model, data_obj, device, best_params['batch_size'])
    print("Predictions generated.")

    data = {"means": mean.tolist(), "errorbars": errorbar.tolist()}
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = 'Submission_' + the_date + '.zip'
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )
    print(f"Submission ZIP saved at: {zip_file}")

if __name__ == '__main__':
    main()


