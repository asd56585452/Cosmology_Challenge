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
import nni
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.strategy import DARTS as DartsStrategy
from nni.nas.experiment import NasExperiment
from nni.nas.evaluator.pytorch import Lightning, Trainer
from nni.nas.evaluator.pytorch.lightning import SupervisedLearningModule
from nni.nas.space import model_context
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
# ### Custom Darts Space

# %%
class CustomDartsSpace(DartsSpace):
    def __init__(self, width=16, num_cells=8, in_channels=1, num_classes=4):
        super().__init__(width=width, num_cells=num_cells, dataset='cifar')

        # Override the first convolution layer to accept 1 input channel
        # We get the out_channels from the original stem convolution
        original_stem_out_channels = self.stem[0].out_channels
        self.stem[0] = nn.Conv2d(in_channels, original_stem_out_channels, 3, 1, 1, bias=False)

        # Override the final classifier for our regression task
        # We get the in_features from the original classifier
        original_classifier_in_features = self.classifier.in_features
        self.classifier = nn.Linear(original_classifier_in_features, num_classes)

# %% [markdown]
# ### NNI Darts Module

# %%
class NniDartsModule(SupervisedLearningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=0.):
        super().__init__(criterion=gaussian_nll_loss, metrics=None,
                         learning_rate=learning_rate, weight_decay=weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = gaussian_nll_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = gaussian_nll_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'loss': loss, 'preds': y_hat, 'labels': y}

    def validation_epoch_end(self, outputs):
        # outputs is a list of dicts from validation_step
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()

        pred_mean = all_preds[:, [0, 2]]
        pred_log_var = all_preds[:, [1, 3]]
        pred_errorbar = np.sqrt(np.exp(pred_log_var))

        val_score = Score._score_phase1(
            true_cosmo=all_labels,
            infer_cosmo=pred_mean,
            errorbar=pred_errorbar
        )
        self.log('val_score', val_score, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

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

def main():
    root_dir = os.getcwd()
    print("Root directory is", root_dir)

    USE_PUBLIC_DATASET = False
    PUBLIC_DATA_DIR = 'public_data/'

    if not USE_PUBLIC_DATASET:
        DATA_DIR = os.path.join(root_dir, 'input_data/')
    else:
        DATA_DIR = PUBLIC_DATA_DIR

    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.mask = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.mask_file)
    if data_obj.USE_PUBLIC_DATASET:
        data_obj.viz_label = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.viz_label_file)
    data_obj.load_test_data()

    N_EPOCHS = 10
    SEARCH_EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    Nsys = data_obj.Nsys
    indices = np.arange(Nsys)
    train_indices, val_indices = train_test_split(indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    kappa_path = os.path.join(DATA_DIR, data_obj.kappa_file)
    label_path = os.path.join(DATA_DIR, data_obj.label_file)

    # DataLoaders for NAS Search Phase
    search_train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
    search_val_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=val_indices, data_obj=data_obj)

    search_train_loader = DataLoader(search_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)
    search_val_loader = DataLoader(search_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=False)

    # 1. NAS Search Phase
    print("Starting NAS Search Phase...")
    model_space = CustomDartsSpace(
        width=8,
        num_cells=4,
        in_channels=1,
        num_classes=4
    )

    evaluator = Lightning(
        NniDartsModule(learning_rate=LEARNING_RATE),
        Trainer(
            max_epochs=SEARCH_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
        ),
        train_dataloaders=search_train_loader,
        val_dataloaders=search_val_loader,
    )

    strategy = DartsStrategy()
    experiment = NasExperiment(model_space, evaluator, strategy)
    experiment.run()

    print("NAS Search Phase completed.")
    exported_arch = experiment.export_top_models(formatter='dict')[0]
    print("Best architecture exported.")

    # 2. NAS Retrain Phase
    print("Starting NAS Retrain Phase...")
    with model_context(exported_arch):
        final_model = CustomDartsSpace(
            width=8,
            num_cells=4,
            in_channels=1,
            num_classes=4
        )

    # DataLoaders for Retraining Phase (using the full training set)
    full_train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=indices, data_obj=data_obj)
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)

    retrain_evaluator = Lightning(
        NniDartsModule(learning_rate=LEARNING_RATE),
        Trainer(
            max_epochs=N_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
        ),
        train_dataloaders=full_train_loader,
        val_dataloaders=search_val_loader, # Can reuse the validation loader
    )

    retrain_evaluator.fit(final_model)
    print("NAS Retrain Phase completed.")

    # 3. Prediction Phase
    print("Generating predictions on the test set...")
    final_model.to(device)
    mean, errorbar = predict(final_model, data_obj, device, BATCH_SIZE)
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


