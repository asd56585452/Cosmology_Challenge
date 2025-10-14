# %% [markdown]
# # FAIR Universe - Weak Lensing ML Uncertainty Challenge
# ## Phase 1 Starting Kit: Power Spectrum Analysis (三階段訓練版本)
# ***
# 
# 此腳本根據使用者需求進行了修改，實現了一個複雜的三階段訓練流程，
# 並使用 Optuna 進行超參數優化。
# 
# **三階段訓練流程 (在每個 Optuna 試驗中):**
# 1.  **Epoch 1-5 (MSE 階段):** 使用 MSE Loss 訓練整個網路，目標是讓模型學會準確預測參數的平均值。
# 2.  **Epoch 6-9 (NLL 階段):** 凍結 CNN 特徵提取層，僅訓練 MLP 分類層。使用 Gaussian NLL Loss，目標是讓模型在準確的平均值基礎上，學習預測合理的不確定性。
# 3.  **Epoch 10 (校準階段):** 凍結 CNN 和 MLP 層，僅訓練兩個新增的 `log_var_scaler` 參數。使用比賽的評分公式作為 Loss，目標是微調不確定性以最大化最終分數。
# 
# ***

# %%
# 0 - 匯入與設定
import os
import json
import time
import zipfile
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import optuna

# %% [markdown]
# # 1 - 輔助類別與函式

# %% [markdown]
# ### 工具函式

# %%
class Utility:
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def load_np(data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        os.makedirs(submission_dir, exist_ok=True)
        json_path = os.path.join(submission_dir, json_file_name)
        with open(json_path, "w") as f:
            json.dump(data, f)
        zip_path = os.path.join(submission_dir, zip_file_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)
        os.remove(json_path)
        return zip_path

# %% [markdown]
# ### 資料處理

# %%
class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'label.npy'
            self.Ncosmo = 101
            self.Nsys = 256
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3
            self.Nsys = 20
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3
        self.shape = [1424, 176]
        self.pixelsize_arcmin = 2
        # 新增 galaxy number density (ng) 的定義
        self.ng = 30

    def load_test_data(self):
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file)
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:, self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.test_kappa_file)

# %% [markdown]
# ### 評分函式

# %%
class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000
        # 加上 epsilon 避免 errorbar 為 0
        epsilon = 1e-8
        score = - np.sum(sq_error / (errorbar**2 + epsilon) + np.log(errorbar**2 + epsilon) + scale_factor * sq_error, 1)
        score = np.mean(score)
        return max(score, -10**6)

# %% [markdown]
# ### PyTorch Dataset

# %%
class WeakLensingDataset(Dataset):
    def __init__(self, kappa_path, label_path, sys_indices, data_obj):
        self.mask = data_obj.mask
        self.shape = data_obj.shape
        self.sys_indices = sys_indices
        self.flat_maps = np.load(kappa_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.Ncosmo = self.labels.shape[0]
        self.Nsys_per_cosmo = len(self.sys_indices)

    def __len__(self):
        return self.Ncosmo * self.Nsys_per_cosmo

    def __getitem__(self, idx):
        cosmo_idx = idx // self.Nsys_per_cosmo
        list_idx = idx % self.Nsys_per_cosmo
        original_sys_idx = self.sys_indices[list_idx]
        data_slice = self.flat_maps[cosmo_idx, original_sys_idx]
        map_data = np.zeros(self.shape, dtype=np.float64)
        map_data[self.mask] = data_slice
        label = self.labels[cosmo_idx, original_sys_idx, :2].astype(np.float32)
        map_tensor = torch.from_numpy(map_data).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float()
        return map_tensor, label_tensor

# %% [markdown]
# ### 模型架構定義 (Vision Transformer)

# %%
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(1424, 176), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 為了讓維度可以整除，在 forward 中進行 padding
        # 計算需要的 padding
        pad_h = (self.patch_size[0] - self.img_size[0] % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - self.img_size[1] % self.patch_size[1]) % self.patch_size[1]

        # apply padding
        x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N) where N = H'*W'
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, hidden_size):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 將 feature extraction 和 classifier 分開，方便凍結
        self.feature_extractor = nn.Sequential(
            self.patch_embed,
            # Transformer a part of feature extractor
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        
        self.log_var_scaler_om = nn.Parameter(torch.zeros(1))
        self.log_var_scaler_s8 = nn.Parameter(torch.zeros(1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, apply_scaler=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 確保 pos_embed 維度正確
        if x.size(1) != self.pos_embed.size(1):
             pos_embed_resized = F.interpolate(
                 self.pos_embed.permute(0, 2, 1),
                 size=x.size(1),
                 mode='linear',
                 align_corners=False
             ).permute(0, 2, 1)
             x = x + pos_embed_resized
        else:
             x = x + self.pos_embed

        x = self.transformer_encoder(x)

        # 從 CLS token 取得輸出
        cls_output = x[:, 0]

        output = self.classifier(cls_output)

        if apply_scaler:
            scaled_output = output.clone()
            scaled_output[:, 1] = output[:, 1] + self.log_var_scaler_om
            scaled_output[:, 3] = output[:, 3] + self.log_var_scaler_s8
            return scaled_output

        return output

# %% [markdown]
# ### 損失函數

# %%
def mse_loss(output, target):
    """只計算平均值 (means) 的 MSE Loss"""
    mean_pred = output[:, [0, 2]]
    mean_target = target[:, :2]
    return nn.functional.mse_loss(mean_pred, mean_target)

def gaussian_nll_loss(output, target):
    """Gaussian Negative Log-Likelihood loss"""
    mean_om, log_var_om = output[:, 0], output[:, 1]
    mean_s8, log_var_s8 = output[:, 2], output[:, 3]
    target_om, target_s8 = target[:, 0], target[:, 1]
    var_om = torch.exp(log_var_om)
    var_s8 = torch.exp(log_var_s8)
    loss_om = 0.5 * (log_var_om + (target_om - mean_om)**2 / var_om)
    loss_s8 = 0.5 * (log_var_s8 + (target_s8 - mean_s8)**2 / var_s8)
    return (loss_om + loss_s8).mean()

def score_phase1_loss(output, target):
    """比賽分數 (_score_phase1) 的 PyTorch 損失函數 (回傳負分數)"""
    infer_cosmo = output[:, [0, 2]]
    log_var = output[:, [1, 3]]
    epsilon = 1e-8
    errorbar_sq = torch.exp(log_var) + epsilon
    true_cosmo = target
    sq_error = (true_cosmo - infer_cosmo)**2
    scale_factor = 1000.0
    score_per_sample = -torch.sum(
        sq_error / errorbar_sq + torch.log(errorbar_sq) + scale_factor * sq_error,
        dim=1
    )
    score = torch.mean(score_per_sample)
    return -score

# %% [markdown]
# ### 預測與雜訊函式

# %%
def add_noise_torch(data, mask, ng, pixel_size=2.):
    noise = torch.randn_like(data) * 0.4 / (2 * ng * pixel_size**2)**0.5
    return data + noise * mask

def predict(model, data_obj, device, batch_size, apply_scaler=False):
    model.eval()
    all_test_preds = []
    test_maps_tensor = torch.from_numpy(data_obj.kappa_test).float().unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_maps_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for maps in test_loader:
            maps = maps[0].to(device)
            outputs = model(maps, apply_scaler=apply_scaler)
            all_test_preds.append(outputs.cpu().numpy())
    all_test_preds = np.concatenate(all_test_preds, axis=0)
    mean = all_test_preds[:, [0, 2]]
    log_var = all_test_preds[:, [1, 3]]
    errorbar = np.sqrt(np.exp(log_var))
    return mean, errorbar

# %% [markdown]
# ### Optuna Objective 函式 (三階段訓練)

# %%
def objective(trial, data_obj, device, mask_tensor, train_indices, fixed_val_dataset, kappa_path, label_path):
    try:
        Utility.set_seed(42)
        # --- 提議超參數 ---
        # 架構
        epochs_stage1 = trial.suggest_int("epochs_stage1", 3, 5)
        epochs_stage2 = trial.suggest_int("epochs_stage2", 3, 5)
        patch_size_factor_h = trial.suggest_categorical("patch_size_factor_h", [8, 16, 32])
        patch_size_factor_w = trial.suggest_categorical("patch_size_factor_w", [8, 11, 16])
        patch_size = (patch_size_factor_h, patch_size_factor_w)

        embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
        depth = trial.suggest_int("depth", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        mlp_ratio = trial.suggest_float("mlp_ratio", 2.0, 4.0)
        hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        
        # 學習率和權重衰減
        lr1 = trial.suggest_float("lr1", 1e-5, 1e-3, log=True)
        wd1 = trial.suggest_float("wd1", 1e-6, 1e-2, log=True)
        lr2 = trial.suggest_float("lr2", 1e-6, 1e-4, log=True)
        wd2 = trial.suggest_float("wd2", 1e-5, 1e-1, log=True)
        lr3 = trial.suggest_float("lr3", 1e-4, 1e-1, log=True)

        # --- 設定 ---
        train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(fixed_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        model = VisionTransformer(
            img_size=tuple(data_obj.shape),
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            hidden_size=hidden_size
        ).to(device)
        total_epochs = epochs_stage1 + epochs_stage2 + 1

        # --- 三階段訓練迴圈 ---
        for epoch in range(total_epochs):
            # --- 決定當前階段 ---
            if epoch < epochs_stage1:
                stage = 1
                loss_fn = mse_loss
                if epoch == 0:
                    optimizer = optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
                    print(f"Trial {trial.number}, Stage 1: Training all params for {epochs_stage1} epochs with MSE Loss.")
            elif epoch < epochs_stage1 + epochs_stage2:
                stage = 2
                loss_fn = gaussian_nll_loss
                if epoch == epochs_stage1: # 進入第二階段時，凍結 ViT 並建立新優化器
                    # for param in model.patch_embed.parameters():
                    #     param.requires_grad = False
                    # for param in model.transformer_encoder.parameters():
                    #     param.requires_grad = False
                    # model.cls_token.requires_grad = False
                    # model.pos_embed.requires_grad = False
                    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2, weight_decay=wd2)
                    optimizer = optim.Adam(model.parameters(), lr=lr2, weight_decay=wd2)
                    print(f"Trial {trial.number}, Stage 2: Training MLP for {epochs_stage2} epochs with NLL Loss.")
            else: # Stage 3
                stage = 3
                loss_fn = score_phase1_loss
                if epoch == epochs_stage1 + epochs_stage2: # 進入第三階段時，凍結 MLP 並建立新優化器
                    for param in model.classifier.parameters():
                        param.requires_grad = False
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr3)
                    print(f"Trial {trial.number}, Stage 3: Training scalers for 1 epoch with Score Loss.")
            
            # --- 訓練 ---
            model.train()
            train_iterator = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{total_epochs} [Stage {stage}]")
            for maps, labels in train_iterator:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                optimizer.zero_grad()
                outputs = model(maps, apply_scaler=(stage == 3))
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # --- 每 Epoch 驗證 ---
            model.eval()
            val_preds_list, val_labels_list = [], []
            with torch.no_grad():
                for maps, labels in val_loader:
                    maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(maps, apply_scaler=(stage == 3))
                    val_preds_list.append(outputs.cpu())
                    val_labels_list.append(labels.cpu())

            all_val_preds = torch.cat(val_preds_list).numpy()
            all_val_labels = torch.cat(val_labels_list).numpy()

            if stage == 1:
                val_loss = nn.functional.mse_loss(torch.from_numpy(all_val_preds[:, [0, 2]]), torch.from_numpy(all_val_labels))
                print(f"Epoch {epoch+1} Val MSE: {val_loss.item():.4f}")
            elif stage == 2:
                val_loss = gaussian_nll_loss(torch.from_numpy(all_val_preds), torch.from_numpy(all_val_labels))
                print(f"Epoch {epoch+1} Val NLL: {val_loss.item():.4f}")
            else: # stage == 3
                pred_mean = all_val_preds[:, [0, 2]]
                pred_log_var = all_val_preds[:, [1, 3]]
                pred_errorbar = np.sqrt(np.exp(pred_log_var))
                val_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
                print(f"Epoch {epoch+1} Val Score: {val_score:.4f}")

        # --- 最終驗證 ---
        model.eval()
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for maps, labels in val_loader:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(maps, apply_scaler=True) # 最終驗證時永遠套用 scaler
                all_val_preds.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
        
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)
        pred_mean = all_val_preds[:, [0, 2]]
        pred_log_var = all_val_preds[:, [1, 3]]
        pred_errorbar = np.sqrt(np.exp(pred_log_var))
        final_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
        
        print(f"Trial {trial.number} Final Score: {final_score:.4f}")
        return final_score

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} failed with CUDA OOM. Returning -1e7.")
            return -1e7 # 回傳一個極差的分數
        else:
            raise e

# %% [markdown]
# # Main

# %%
def main():
    Utility.set_seed(42)
    root_dir = os.getcwd()
    USE_PUBLIC_DATASET = True
    DATA_DIR = 'public_data/' if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')
    N_TRIALS = 1000
    N_JOBS = 1
    TIMEOUT = 3600 * 12

    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.load_test_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- 準備訓練/驗證資料 ---
    kappa_path = os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa.npy')
    label_path = os.path.join(DATA_DIR, 'label.npy')
    all_labels = np.load(label_path)
    Nsys = all_labels.shape[1]
    indices = np.arange(Nsys)
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    print("Generating a fixed noisy validation set...")
    val_dataset_for_noise = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=val_indices, data_obj=data_obj)
    val_loader_for_noise = DataLoader(val_dataset_for_noise, batch_size=32, shuffle=False)
    noisy_val_maps_list, val_labels_list = [], []
    with torch.no_grad():
        for maps, labels in tqdm(val_loader_for_noise, desc="Pre-noising validation data"):
            maps = maps.to(device)
            noisy_maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            noisy_val_maps_list.append(noisy_maps.cpu())
            val_labels_list.append(labels.cpu())
    noisy_val_maps_tensor = torch.cat(noisy_val_maps_list)
    val_labels_tensor = torch.cat(val_labels_list)
    fixed_val_dataset = torch.utils.data.TensorDataset(noisy_val_maps_tensor, val_labels_tensor)
    print("Fixed noisy validation set generated.")

    # --- Optuna 參數搜索 ---
    study = optuna.create_study(
        study_name="weak_lensing_3_stage_v2",
        storage="sqlite:///optuna_study_3_stage_v2.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(
        lambda trial: objective(trial, data_obj, device, mask_tensor, train_indices, fixed_val_dataset, kappa_path, label_path),
        n_jobs=N_JOBS,
        n_trials=N_TRIALS,
        timeout=TIMEOUT
    )
    
    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params
    with open("best_hyperparameters_3_stage_v2.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # --- 使用最佳參數進行最終訓練 ---
    Utility.set_seed(42)
    print("\n" + "="*20 + " Final Training with Best Params " + "="*20)
    
    # 準備資料
    train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(fixed_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 建立模型
    model = VisionTransformer(
        img_size=tuple(data_obj.shape),
        patch_size=(best_params['patch_size_factor_h'], best_params['patch_size_factor_w']),
        in_chans=1,
        embed_dim=best_params['embed_dim'],
        depth=best_params['depth'],
        num_heads=best_params['num_heads'],
        mlp_ratio=best_params['mlp_ratio'],
        hidden_size=best_params['hidden_size']
    ).to(device)
    
    epochs_stage1 = best_params['epochs_stage1']
    epochs_stage2 = best_params['epochs_stage2']
    total_epochs = epochs_stage1 + epochs_stage2 + 1
    best_val_score = -np.inf
    
    # 完全重現三階段訓練流程
    for epoch in range(total_epochs):
        if epoch < epochs_stage1:
            stage, loss_fn = 1, mse_loss
            if epoch == 0:
                optimizer = optim.Adam(model.parameters(), lr=best_params['lr1'], weight_decay=best_params['wd1'])
        elif epoch < epochs_stage1 + epochs_stage2:
            stage, loss_fn = 2, gaussian_nll_loss
            if epoch == epochs_stage1:
                for param in model.patch_embed.parameters():
                    param.requires_grad = False
                for param in model.transformer_encoder.parameters():
                    param.requires_grad = False
                model.cls_token.requires_grad = False
                model.pos_embed.requires_grad = False
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params['lr2'], weight_decay=best_params['wd2'])
        else:
            stage, loss_fn = 3, score_phase1_loss
            if epoch == epochs_stage1 + epochs_stage2:
                for param in model.classifier.parameters(): param.requires_grad = False
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params['lr3'])

        # 訓練
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{total_epochs} [Stage {stage}]")
        for maps, labels in train_iterator:
            maps, labels = maps.to(device), labels.to(device)
            maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            optimizer.zero_grad()
            outputs = model(maps, apply_scaler=(stage == 3))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # 每 Epoch 驗證
        model.eval()
        val_preds_list, val_labels_list = [], []
        with torch.no_grad():
            for maps, labels in val_loader:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(maps, apply_scaler=(stage == 3))
                val_preds_list.append(outputs.cpu())
                val_labels_list.append(labels.cpu())

        all_val_preds = torch.cat(val_preds_list).numpy()
        all_val_labels = torch.cat(val_labels_list).numpy()

        if stage == 1:
            val_loss = nn.functional.mse_loss(torch.from_numpy(all_val_preds[:, [0, 2]]), torch.from_numpy(all_val_labels))
            print(f"Epoch {epoch+1} Val MSE: {val_loss.item():.4f}")
        elif stage == 2:
            val_loss = gaussian_nll_loss(torch.from_numpy(all_val_preds), torch.from_numpy(all_val_labels))
            print(f"Epoch {epoch+1} Val NLL: {val_loss.item():.4f}")
        else: # stage == 3
            pred_mean = all_val_preds[:, [0, 2]]
            pred_log_var = all_val_preds[:, [1, 3]]
            pred_errorbar = np.sqrt(np.exp(pred_log_var))
            val_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
            print(f"Epoch {epoch+1} Val Score: {val_score:.4f}")

    # 最終驗證與儲存
    print(f"Final Scalers -> om: {model.log_var_scaler_om.item():.4f}, s8: {model.log_var_scaler_s8.item():.4f}")
    torch.save(model.state_dict(), 'best_model_3_stage_v2.pth')
    print("Final model saved to best_model_3_stage_v2.pth")

    # --- 產生提交檔案 ---
    print("\nGenerating predictions on the test set...")
    mean, errorbar = predict(model, data_obj, device, best_params['batch_size'], apply_scaler=True)
    print("Predictions generated.")
    
    data = {"means": mean.tolist(), "errorbars": errorbar.tolist()}
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = f'Submission_{the_date}_3_stage.zip'
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )
    print(f"Submission ZIP saved at: {zip_file}")


if __name__ == '__main__':
    main()

