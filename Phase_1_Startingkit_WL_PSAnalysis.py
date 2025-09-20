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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import shutil
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
# ### Visualization

# %%
class Visualization:

    @staticmethod
    def plot_mask(mask):
        plt.figure(figsize=(30,100))
        plt.imshow(mask.T)
        plt.show()

    @staticmethod
    def plot_noiseless_training_convergence_map(kappa):
        plt.figure(figsize=(30,100))
        plt.imshow(kappa[0,0].T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_noisy_training_convergence_map(kappa, mask, pixelsize_arcmin, ng):
        plt.figure(figsize=(30,100))
        plt.imshow(Utility.add_noise(kappa[0,0], mask, ng, pixelsize_arcmin).T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_cosmological_parameters_OmegaM_S8(label):
        plt.scatter(label[:,0,0], label[:,0,1])
        plt.xlabel(r'$\Omega_m$')
        plt.ylabel(r'$S_8$')
        plt.show()

    @staticmethod
    def plot_baryonic_physics_parameters(label):
        plt.scatter(label[0,:,2], label[0,:,3])
        plt.xlabel(r'$T_{\mathrm{AGN}}$')
        plt.ylabel(r'$f_0$')
        plt.show()

    @staticmethod
    def plot_photometric_redshift_uncertainty_parameters(label):
        plt.hist(label[0,:,4], bins=20)
        plt.xlabel(r'$\Delta z$')
        plt.show()

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
import torch
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional

from torch import nn, Tensor
from torch.nn import functional as F

# The following is copied from torchvision.ops.misc
class Permute(nn.Module):
    """
    This class is adapted from a private class in torchvision.ops.misc
    """

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)

# The following is copied from torchvision.ops.stochastic_depth
class StochasticDepth(nn.Module):
    """
    This class is adapted from a private class in torchvision.ops.stochastic_depth
    """
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if self.mode == "row":
            noise = torch.rand(batch_size, 1, 1, 1, device=device, dtype=dtype)
        else:
            noise = torch.rand(1, device=device, dtype=dtype)

        survival_mask = noise > self.p
        x = x.div(1.0 - self.p) if self.training else x
        return x * survival_mask

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 4, # Changed from 1000 to 4
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: list[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(1, firstconv_output_channels, kernel_size=4, stride=4, padding=0, bias=True),
                norm_layer(firstconv_output_channels)
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def convnext_tiny_custom(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

def convnext_pico_custom(**kwargs: Any) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(48, 96, 3),
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 9),
        CNBlockConfig(384, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

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

def predict(model_paths, data_obj, device, batch_size):
    all_model_preds = []

    test_maps_tensor = torch.from_numpy(data_obj.kappa_test).float().unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_maps_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for model_path in model_paths:
        print(f"Loading model: {model_path}")
        model = convnext_pico_custom().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        current_model_preds = []
        with torch.no_grad():
            for maps in test_loader:
                maps = maps[0].to(device)
                outputs = model(maps)
                current_model_preds.append(outputs.cpu().numpy())

        all_model_preds.append(np.concatenate(current_model_preds, axis=0))

    # Stack predictions along a new axis (models, samples, outputs)
    all_model_preds = np.stack(all_model_preds)

    # all_model_preds has shape (N_ENSEMBLE, N_test, 4)
    return all_model_preds

def main():
    root_dir = os.getcwd()
    print("Root directory is", root_dir)

    # %% [markdown]
    # **Option 1: To quickly run through this starting kit:** You may set `USE_PUBLIC_DATASET = False` so that only a downsampled training data and test data will be loaded. In the downsampled training data, there are $N_{\rm cosmo}=3$ cosmological models and $N_{\rm sys}=20$ realizations of nuisance parameters. The downsampled test data contains $N_{\rm test}=3$ instances.
    #
    # ***
    #
    # #### ⚠️ NOTE:
    # To make a valid submission and obtain a score on Codabench, **you will need to make your predictions using the entire test data set (4,000 instances). Use the Option 2 below to load the entire test data.**
    #
    # ***
    #
    # **Option 2: To load the entire training data and test data:** Set `USE_PUBLIC_DATASET = True` and specify a path where you will save the downloaded public data from Codabench. In the entire training data, there are $N_{\rm cosmo}=101$ cosmological models and $N_{\rm sys}=256$ realizations of nuisance parameters. The entire test data contains $N_{\rm test}=4000$ instances.

    # %%
    USE_PUBLIC_DATASET = True

    # USE_PUBLIC_DATASET = True
    PUBLIC_DATA_DIR = 'public_data/'  # This is only required when you set USE_PUBLIC_DATASET = True

    # %%
    if not USE_PUBLIC_DATASET:                                         # Testing this startking kit with a tiny sample of the training data (3, 20, 1424, 176)
        DATA_DIR = os.path.join(root_dir, 'input_data/')
    else:                                                              # Training your model with all training data (101, 256, 1424, 176)
        DATA_DIR = PUBLIC_DATA_DIR

    # %% [markdown]
    # ### Load the train and test data

    # %%
    # Initialize Data class object
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)

    # Load only the mask and test data in the main process. The training data will be
    # memory-mapped by the Dataset workers.
    data_obj.mask = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.mask_file)
    if data_obj.USE_PUBLIC_DATASET: # The viz_label is only available for the public dataset
        data_obj.viz_label = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.viz_label_file)
    data_obj.load_test_data()

    # %% [markdown]
    # #### ⚠️ NOTE:
    # - The original training images are *noiseless* (without any pixel-level noise).
    # - The original test images is *noisy* (pixel-level noise with galaxy number density $n_g = 30~\text{arcmin}^{-2}$ and pixel size $=2$ arcmin has been added).
    #
    # - **You will have to add pixel-level noise to the training data through the helper function** `Utility.add_noise`.
    #
    # For example:
    # ```python
    #     noisy_kappa = Utility.add_noise(data=data_obj.kappa.astype(np.float64),
    #                                     mask=data_obj.mask,
    #                                     ng=data_obj.ng,
    #                                     pixel_size=data_obj.pixelsize_arcmin)
    # ```
    #
    # The shape of `noisy_kappa` will be the same as the shape of `data_obj.kappa`.

    # %%
    Ncosmo = data_obj.Ncosmo
    Nsys = data_obj.Nsys

    print(f'There are {Ncosmo} cosmological models, each has {Nsys} realizations of nuisance parameters in the training data.')

    # %%
    print(f'Shape of the mask = {data_obj.mask.shape}')
    print(f'Shape of the test data = {data_obj.kappa_test.shape}')

    # %% [markdown]
    # #### ⚠️ NOTE:
    #
    # If you want to split your own training/validation sets to evaluate your model, we recommend splitting the original training data along `axis = 1` (the 256 realizations of nuisance parameters). This will ensure that there are no intrinsic correlations between the training and validation sets.
    #
    # For example, you may split the data by the following script:
    # ```python
    #     from sklearn.model_selection import train_test_split
    #
    #     NP_idx = np.arange(Nsys)  # The indices of Nsys nuisance parameter realizations
    #     split_fraction = 0.2      # Set the fraction of data you want to split (between 0 and 1)
    #     seed = 5566               # Define your random seed for reproducible results
    #
    #     train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=split_fraction,
    #                                                 random_state=seed)
    #
    #     kappa_train = data_obj.kappa[:, train_NP_idx]         # shape = (Ncosmo, Ntrain, 1424, 176)
    #     label_train = data_obj.label[:, train_NP_idx]         # shape = (Ncosmo, Ntrain, 5)
    #     kappa_val = data_obj.kappa[:, val_NP_idx]             # shape = (Ncosmo, Nval, 1424, 176)
    #     label_val = data_obj.label[:, val_NP_idx]             # shape = (Ncosmo, Nval, 5)
    # ```

    # %% [markdown]
    # # 3 - Visualization

    # %% [markdown]
    # ### 2D training maps

    # %% [markdown]
    # survey mask: a binary map that shows which parts of the sky are observed (yellow) and which areas are blocked (purple)

    # %%
    # mask
    # Visualization.plot_mask(mask=data_obj.mask)

    # %% [markdown]
    # noiseless training convergence map: The convergence maps show the projected matter density (including dark matter and ordinary matter) in the simulated universe, under the Born approximation. On large scales, we can see the matter forms web-like structures (cosmic web) in the universe. The dense regions in these maps, called dark matter halos, are the sites where galaxies form and reside.

    # %%
    # # noiseless training convergence map
    # Visualization.plot_noiseless_training_convergence_map(kappa=data_obj.kappa)

    # %% [markdown]
    # noisy training convergence map: We add Gaussian noise to the data. This mimics the observed data. During training the noise can be added on the fly with different realizations.

    # %%
    # noisy training convergence map
    # Visualization.plot_noisy_training_convergence_map(kappa=data_obj.kappa,
    #                                                   mask=data_obj.mask,
    #                                                   pixelsize_arcmin=data_obj.pixelsize_arcmin,
    #                                                   ng=data_obj.ng)

    # %% [markdown]
    # ### Distribution of physical parameters

    # %% [markdown]
    # Distribution of cosmological parameters $\Omega_m$ and $S_8$. The density increases towards fiducial cosmology. Note that this distribution introduces a prior in the analysis. The test data cosmology follows the same distribution as the training data.

    # %%
    # Visualization.plot_cosmological_parameters_OmegaM_S8(label=data_obj.viz_label)

    # %% [markdown]
    # Distribution of baryonic physics parameters. These are nuisance parameters and should be marginalized in the analysis. They follow a uniform distribution within the prior range $T_{\mathrm{AGN}} \in [7.2, 8.5]$, $f_0 \in [0, 0.0265]$

    # %%
    # Visualization.plot_baryonic_physics_parameters(label=data_obj.viz_label)

    # %% [markdown]
    # Distribution of photometric redshift uncertainty parameters. This is a nuisance parameter and should be marginalized in the analysis. It follows a Gaussian distribution with mean 0 and std 0.022

    # %%
    # Visualization.plot_photometric_redshift_uncertainty_parameters(label=data_obj.viz_label)

    # %% [markdown]
    # # 5 - Phase one inference
    # We will now train the CNN emulator.

    # %%
    # -- Hyperparameters --
    N_SPLITS = 5 # For 5-fold cross-validation
    N_EPOCHS = 30 # A reasonable default. User may need to adjust for full training runs.
    BATCH_SIZE = 16 # Reduced batch size to prevent OOM error
    LEARNING_RATE = 1e-4
    RANDOM_SEED = 42

    # -- Device Setup --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move mask to device and add batch/channel dimensions for broadcasting
    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # -- Data Splitting with K-Fold --
    Nsys = data_obj.Nsys
    indices = np.arange(Nsys)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Define paths to the full, original data files
    kappa_path = os.path.join(DATA_DIR, data_obj.kappa_file)
    label_path = os.path.join(DATA_DIR, data_obj.label_file)

    # --- K-Fold Cross-Validation Training Loop ---
    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")

        # Set a different seed for each fold to ensure different weight initializations
        torch.manual_seed(RANDOM_SEED + fold)
        np.random.seed(RANDOM_SEED + fold)

        # -- Create Datasets and DataLoaders for the current fold --
        train_dataset = WeakLensingDataset(
            kappa_path=kappa_path,
            label_path=label_path,
            sys_indices=indices[train_indices],
            data_obj=data_obj,
            train=True
        )
        val_dataset = WeakLensingDataset(
            kappa_path=kappa_path,
            label_path=label_path,
            sys_indices=indices[val_indices],
            data_obj=data_obj,
            train=True # Add noise to validation as well
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,persistent_workers=True)

        # -- Model, Optimizer --
        model = convnext_pico_custom().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS * len(train_loader))

        # -- Training Loop for one fold --
        best_val_score = -np.inf
        model_path = f'best_model_fold_{fold}.pth'

        for epoch in range(N_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            train_iterator = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{N_EPOCHS} [Train]")
            for maps, labels in train_iterator:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                optimizer.zero_grad()
                outputs = model(maps)
                loss = gaussian_nll_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * maps.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                val_iterator = tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{N_EPOCHS} [Val]")
                for maps, labels in val_iterator:
                    maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                    outputs = model(maps)
                    loss = gaussian_nll_loss(outputs, labels)
                    val_loss += loss.item() * maps.size(0)
                    all_val_preds.append(outputs.cpu().numpy())
                    all_val_labels.append(labels.cpu().numpy())
            val_loss /= len(val_loader.dataset)

            # Calculate validation score
            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            pred_mean = all_val_preds[:, [0, 2]]
            pred_log_var = all_val_preds[:, [1, 3]]
            pred_errorbar = np.sqrt(np.exp(pred_log_var))
            val_score = Score._score_phase1(
                true_cosmo=all_val_labels,
                infer_cosmo=pred_mean,
                errorbar=pred_errorbar
            )

            print(f"Fold {fold+1} Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                print(f"  New best score for fold {fold+1}: {best_val_score:.4f}. Saving model to {model_path}...")
                torch.save(model.state_dict(), model_path)

    # --- Ensemble Prediction and Aggregation ---
    print("\n--- Generating and Aggregating Ensemble Predictions ---")

    model_paths = [f'best_model_fold_{i}.pth' for i in range(N_SPLITS)]

    # Get predictions from all models: shape (N_ENSEMBLE, N_test, 4)
    all_preds = predict(model_paths, data_obj, device, BATCH_SIZE)

    # Separate means and log_vars
    # means_all_models has shape (N_ENSEMBLE, N_test, 2)
    means_all_models = all_preds[:, :, [0, 2]]
    # log_vars_all_models has shape (N_ENSEMBLE, N_test, 2)
    log_vars_all_models = all_preds[:, :, [1, 3]]

    # 1. Calculate final ensemble mean
    # ens_mean has shape (N_test, 2)
    ens_mean = np.mean(means_all_models, axis=0)

    # 2. Calculate final ensemble variance
    # First, get aleatoric variance for each model
    aleatoric_vars = np.exp(log_vars_all_models) # shape (N_ENSEMBLE, N_test, 2)

    # Average aleatoric variance across models
    # avg_aleatoric_var has shape (N_test, 2)
    avg_aleatoric_var = np.mean(aleatoric_vars, axis=0)

    # Calculate epistemic variance (variance of the means)
    # epistemic_var has shape (N_test, 2)
    epistemic_var = np.var(means_all_models, axis=0)

    # Total variance is the sum of the two
    total_variance = avg_aleatoric_var + epistemic_var

    # The errorbar for submission is the standard deviation
    ens_errorbar = np.sqrt(total_variance)

    print("Ensemble predictions generated and aggregated.")

    # %% [markdown]
    # #### ⚠️ NOTE:
    # - `mean`: a 2D array containing the point estimates of 2 cosmological parameters $\hat{\Omega}_m$ and $\hat{S}_8$.
    # - `errorbar`: a 2D array containing the one-standard deviation uncertainties of 2 cosmological parameters $\hat{\sigma}_{\Omega_m}$ and  $\hat{\sigma}_{S_8}$.
    #
    # The shapes of `mean`, and `errorbar` must be $(N_{\rm test}, 2)$.
    #
    # ***

    # %% [markdown]
    # # 6 - (Optional) Prepare submission for Codabench

    # %% [markdown]
    # ***
    #
    # This section will save the model predictions `mean` and `errorbar` (both are 2D arrays with shape `(4000, 2)`, where `4000` is the number of test instances and `2` is the number of our parameters of interest) as a dictionary in a JSON file `result.json`. Then it will compress `result.json` into a zip file that can be directly submitted to Codabench.
    #
    # ***

    # %%
    data = {"means": ens_mean.tolist(), "errorbars": ens_errorbar.tolist()}
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


