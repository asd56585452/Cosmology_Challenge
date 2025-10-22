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
# ### 模型架構定義 (DynamicCNN)

# %%
# --- 先定義好注意力模組 ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

# --- 接著，定義包含殘差連接的注意力區塊 ---
class ResAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResAttentionBlock, self).__init__()

        # 主要路徑
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 注意力機制
        self.se = SEBlock(out_channels)
        self.sa = SpatialAttention()

        # 捷徑 (Shortcut / Residual Connection)
        self.shortcut = nn.Sequential()
        # 如果維度不匹配 (輸入/輸出通道數不同，或步長>1導致尺寸變化)，則需要用 1x1 卷積來匹配維度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主要路徑計算
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 應用注意力
        out = self.se(out)
        out = self.sa(out)

        # 加入殘差連接
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResAttentionCNN(nn.Module):
    # 在建構子中加入 base_channels 參數
    def __init__(self, layer_counts, base_channels, hidden_size, dropout_rate=0.5):
        super(ResAttentionCNN, self).__init__()

        # 初始通道數現在是動態的
        self.in_channels = base_channels

        # 初始卷積層 (使用 base_channels)
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()

        # --- 動態建立 block_configs ---
        # 規則：每經過一個 stride=2 的 block，通道數加倍
        current_channels = base_channels
        block_configs = []

        # 假設我們支援的最大深度是 6
        max_supported_blocks = 6
        for i in range(max_supported_blocks):
            if i == 0: # Block 1
                stride = 1
                # Block 1 通道數 = base_channels
            else: # Block 2, 3, 4...
                stride = 2
                current_channels *= 2 # 通道數加倍

            block_configs.append((current_channels, stride))

        # --- 根據傳入的 layer_counts (n_blocks) 來建立 ---
        n_blocks = len(layer_counts)
        for i in range(n_blocks):
            num_layers_in_block = layer_counts[i]
            out_channels, stride = block_configs[i] # 從動態 config 讀取

            block = self._make_layer(out_channels, num_layers_in_block, stride)
            self.layers.append(block)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分類器的輸入通道數 = 最後一個 block 的輸出通道數
        final_out_channels = self.in_channels

        self.classifier = nn.Sequential(
            nn.Linear(final_out_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 4)
        )
        
    def _make_layer(self, out_channels, num_blocks, stride):
        # ... (與前次回答中的 _make_layer 程式碼相同)
        if num_blocks == 0:
            if stride == 1 and self.in_channels == out_channels:
                 return nn.Identity()
            else:
                 shortcut_only_block = nn.Sequential(
                     nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(out_channels)
                 )
                 self.in_channels = out_channels
                 return shortcut_only_block
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResAttentionBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # ... (與前次回答中的 forward 程式碼相同)
        out = self.relu(self.bn1(self.conv1(x)))
        for layer_block in self.layers:
            out = layer_block(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out

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

def predict(model, data_obj, device, batch_size):
    model.eval()
    all_test_preds = []
    test_maps_tensor = torch.from_numpy(data_obj.kappa_test).float().unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_maps_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for maps in test_loader:
            maps = maps[0].to(device)
            outputs = model(maps)
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
        max_epochs_stage1 = 6
        max_epochs_stage2 = 5
        epochs_stage1 = trial.suggest_int("epochs_stage1", 3, max_epochs_stage1)
        epochs_stage2 = trial.suggest_int("epochs_stage2", 2, max_epochs_stage2)

        # 架構
        base_channels = trial.suggest_categorical("base_channels", [8, 10, 14, 22])
        n_blocks = trial.suggest_int("n_blocks", 3, 5)
        layer_counts = [trial.suggest_int(f"block_{i+1}_layers", 0, 3) for i in range(n_blocks)]
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        
        # 學習率和權重衰減
        lr1 = trial.suggest_float("lr1", 1e-5, 1e-3, log=True)
        wd1 = trial.suggest_float("wd1", 1e-6, 1e-2, log=True)
        lr2 = trial.suggest_float("lr2", 1e-6, 1e-4, log=True)
        wd2 = trial.suggest_float("wd2", 1e-5, 1e-1, log=True)

        # --- 設定 ---
        train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(fixed_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        model = ResAttentionCNN(
            layer_counts=layer_counts,
            base_channels=base_channels,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        ).to(device)

        total_epochs = epochs_stage1 + epochs_stage2

        # --- 兩階段訓練迴圈 ---
        for epoch in range(total_epochs):
            # --- 決定當前階段 ---
            if epoch < epochs_stage1:
                stage = 1
                loss_fn = mse_loss
                if epoch == 0:
                    optimizer = optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
                    print(f"Trial {trial.number}, Stage 1: Training all params with MSE Loss for {epochs_stage1} epochs.")
            else: # Stage 2
                stage = 2
                loss_fn = gaussian_nll_loss
                if epoch == epochs_stage1: # 進入第二階段時
                    # Freeze feature extractor layers
                    for name, param in model.named_parameters():
                        if 'classifier' not in name and 'log_var_scaler' not in name:
                            param.requires_grad = False
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2, weight_decay=wd2)
                    print(f"Trial {trial.number}, Stage 2: Training Classifier with NLL Loss for {epochs_stage2} epochs.")
            
            # --- 訓練 ---
            model.train()
            train_iterator = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{total_epochs} [Stage {stage}]")
            for maps, labels in train_iterator:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                optimizer.zero_grad()
                outputs = model(maps)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # --- 每 Epoch 驗證 ---
            model.eval()
            all_val_preds, all_val_labels = [], []
            with torch.no_grad():
                for maps, labels in val_loader:
                    maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(maps)
                    all_val_preds.append(outputs.cpu().numpy())
                    all_val_labels.append(labels.cpu().numpy())
            
            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)

            # 根據當前階段選擇評估指標
            if stage == 1:
                pred_mean = all_val_preds[:, [0, 2]]
                val_metric = -nn.functional.mse_loss(torch.from_numpy(pred_mean), torch.from_numpy(all_val_labels)).item()
                metric_name = "Val MSE"
                print(f"Epoch {epoch+1}/{total_epochs} - {metric_name}: {val_metric:.4f}")
            else: # Stage 2
                pred_mean = all_val_preds[:, [0, 2]]
                pred_log_var = all_val_preds[:, [1, 3]]
                pred_errorbar = np.sqrt(np.exp(pred_log_var))
                val_metric = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
                metric_name = "Val Score"
                print(f"Epoch {epoch+1}/{total_epochs} - {metric_name}: {val_metric:.4f}")
            # trial.report(val_metric, epoch-epochs_stage1)

            # # --- Early Stopping ---
            # if trial.should_prune():
            #     print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            #     raise optuna.exceptions.TrialPruned()
        
        # --- 最終回傳分數 ---
        final_score = val_metric
        print(f"Trial {trial.number} Final Score: {final_score:.4f}")
        return final_score

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} failed with CUDA OOM. Pruning trial.")
            raise optuna.exceptions.TrialPruned()
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
    TIMEOUT = 3600 * 24 * 2

    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.load_test_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- 準備訓練/驗證資料 ---
    kappa_path = os.path.join(DATA_DIR, data_obj.kappa_file)
    label_path = os.path.join(DATA_DIR, data_obj.label_file)
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
        study_name="weak_lensing_2_stage_resnet",
        storage="sqlite:///optuna_study_2_stage_resnet.db",
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
    with open("best_hyperparameters_2_stage_resnet.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # --- 使用最佳參數進行最終訓練 ---
    Utility.set_seed(42)
    print("\n" + "="*20 + " Final Training with Best Params " + "="*20)
    
    # 準備資料
    train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(fixed_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 從 best_params 重建 layer_counts
    n_blocks = best_params['n_blocks']
    layer_counts = [best_params[f'block_{i+1}_layers'] for i in range(n_blocks)]

    # 建立模型
    model = ResAttentionCNN(
        layer_counts=layer_counts,
        base_channels=best_params['base_channels'],
        hidden_size=best_params['hidden_size'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    # 從 best_params 獲取 epoch 數
    epochs_stage1 = best_params.get('epochs_stage1', 5)
    epochs_stage2 = best_params.get('epochs_stage2', 4)
    total_epochs = epochs_stage1 + epochs_stage2
    
    # 重現兩階段訓練流程
    for epoch in range(total_epochs):
        if epoch < epochs_stage1:
            stage, loss_fn = 1, mse_loss
            if epoch == 0:
                optimizer = optim.Adam(model.parameters(), lr=best_params['lr1'], weight_decay=best_params['wd1'])
        else: # Stage 2
            stage, loss_fn = 2, gaussian_nll_loss
            if epoch == epochs_stage1:
                for name, param in model.named_parameters():
                    if 'classifier' not in name and 'log_var_scaler' not in name:
                        param.requires_grad = False
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=best_params['lr2'], weight_decay=best_params['wd2'])

        # 訓練
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{total_epochs} [Stage {stage}]")
        for maps, labels in train_iterator:
            maps, labels = maps.to(device), labels.to(device)
            maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            optimizer.zero_grad()
            outputs = model(maps)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # 最終驗證與儲存
    model.eval()
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for maps, labels in val_loader:
            maps, labels = maps.to(device), labels.to(device)
            outputs = model(maps)
            all_val_preds.append(outputs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    pred_mean = all_val_preds[:, [0, 2]]
    pred_log_var = all_val_preds[:, [1, 3]]
    pred_errorbar = np.sqrt(np.exp(pred_log_var))
    final_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
    
    print(f"Final Model Validation Score: {final_score:.4f}")
    torch.save(model.state_dict(), 'best_model_2_stage_resnet.pth')
    print("Final model saved to best_model_2_stage_resnet.pth")

    # --- 產生提交檔案 ---
    print("\nGenerating predictions on the test set...")
    mean, errorbar = predict(model, data_obj, device, best_params['batch_size'])
    print("Predictions generated.")
    
    data = {"means": mean.tolist(), "errorbars": errorbar.tolist()}
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_file_name = f'Submission_{the_date}_2_stage_resnet.zip'
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )
    print(f"Submission ZIP saved at: {zip_file}")


if __name__ == '__main__':
    main()

