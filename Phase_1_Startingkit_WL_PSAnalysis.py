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
class DynamicCNN(nn.Module):
    def __init__(self, nf_scalings, layer_counts, hidden_size):
        super(DynamicCNN, self).__init__()
        nf = 8
        features = nn.ModuleList()
        in_c = 1
        for i in range(len(layer_counts)):
            out_c = int(nf * (2 ** i) * nf_scalings[i])
            features.append(nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()))
            for _ in range(layer_counts[i] - 1):
                features.append(nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()))
            in_c = out_c
            if i < len(layer_counts) - 1:
                features.append(nn.AvgPool2d(2, 2))
        
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 修改分類器以包含一個可調整的隱藏層
        self.classifier = nn.Sequential(
            nn.Linear(in_c, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        
        # 新增兩個獨立的可學習縮放參數
        self.log_var_scaler_om = nn.Parameter(torch.zeros(1))
        self.log_var_scaler_s8 = nn.Parameter(torch.zeros(1))

    def forward(self, x, apply_scaler=False):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        if apply_scaler:
            output = x.clone()
            # 分別套用縮放參數
            output[:, 1] = x[:, 1] + self.log_var_scaler_om
            output[:, 3] = x[:, 3] + self.log_var_scaler_s8
            return output
        
        return x

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
        # 新增: 調整前兩階段的 Epoch 數量
        epochs_stage1 = trial.suggest_int("epochs_stage1", 3, 6)
        epochs_stage2 = trial.suggest_int("epochs_stage2", 2, 5)

        # 架構
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        nf_scalings = [trial.suggest_float(f"block_{i}_nf_scaling", 0.25, 4, log=True) for i in range(6)]
        layer_counts = [trial.suggest_int(f"block_{i}_layers", 0, 4) for i in range(6)]
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
        
        model = DynamicCNN(nf_scalings=nf_scalings, layer_counts=layer_counts, hidden_size=hidden_size).to(device)
        # 總 Epoch 數現在是動態的
        total_epochs = epochs_stage1 + epochs_stage2 + 1 # 加上第三階段的 1 個 epoch

        # --- 三階段訓練迴圈 ---
        for epoch in range(total_epochs):
            # --- 決定當前階段 ---
            if epoch < epochs_stage1:
                stage = 1
                loss_fn = mse_loss
                if epoch == 0:
                    optimizer = optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
                    print(f"Trial {trial.number}, Stage 1: Training all params with MSE Loss for {epochs_stage1} epochs.")
            elif epoch < epochs_stage1 + epochs_stage2:
                stage = 2
                loss_fn = gaussian_nll_loss
                if epoch == epochs_stage1: # 進入第二階段時
                    for param in model.features.parameters():
                        param.requires_grad = False
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2, weight_decay=wd2)
                    print(f"Trial {trial.number}, Stage 2: Training MLP with NLL Loss for {epochs_stage2} epochs.")
            else: # 最後一個 epoch
                stage = 3
                loss_fn = score_phase1_loss
                if epoch == epochs_stage1 + epochs_stage2: # 進入第三階段時
                    for param in model.classifier.parameters():
                        param.requires_grad = False
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr3)
                    print(f"Trial {trial.number}, Stage 3: Training scalers with Score Loss for 1 epoch.")
            
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
            all_val_preds, all_val_labels = [], []
            with torch.no_grad():
                for maps, labels in val_loader:
                    maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    # 根據當前階段決定是否套用 scaler
                    outputs = model(maps, apply_scaler=(stage >= 3)) 
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
            else:
                val_metric = -gaussian_nll_loss(torch.from_numpy(all_val_preds), torch.from_numpy(all_val_labels)).item()
                metric_name = "Val NLL"
                print(f"Epoch {epoch+1}/{total_epochs} - {metric_name}: {val_metric:.4f}")
                pred_mean = all_val_preds[:, [0, 2]]
                pred_log_var = all_val_preds[:, [1, 3]]
                pred_errorbar = np.sqrt(np.exp(pred_log_var))
                val_metric = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
                metric_name = "Val Score"
                print(f"Epoch {epoch+1}/{total_epochs} - {metric_name}: {val_metric:.4f}")
        
        # --- 最終回傳分數 ---
        # Optuna 會以最後一個 epoch 的分數作為最終目標
        final_score = val_metric
        print(f"Trial {trial.number} Final Score: {final_score:.4f}")
        return final_score

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} failed with CUDA OOM. Pruning trial.")
            # 使用 Optuna 的剪枝機制
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
    TIMEOUT = 3600 * 20

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
        study_name="weak_lensing_3_stage",
        storage="sqlite:///optuna_study_3_stage.db",
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
    with open("best_hyperparameters_3_stage.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # --- 使用最佳參數進行最終訓練 ---
    Utility.set_seed(42)
    print("\n" + "="*20 + " Final Training with Best Params " + "="*20)
    
    # 準備資料
    train_dataset = WeakLensingDataset(kappa_path=kappa_path, label_path=label_path, sys_indices=train_indices, data_obj=data_obj)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(fixed_val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # 建立模型
    model = DynamicCNN(
        nf_scalings=[best_params[f'block_{i}_nf_scaling'] for i in range(6)],
        layer_counts=[best_params[f'block_{i}_layers'] for i in range(6)],
        hidden_size=best_params['hidden_size']
    ).to(device)
    
    # 從 best_params 獲取 epoch 數
    epochs_stage1 = best_params.get('epochs_stage1', 5) # 提供預設值以防舊的存檔沒有這個參數
    epochs_stage2 = best_params.get('epochs_stage2', 4)
    total_epochs = epochs_stage1 + epochs_stage2 + 1
    
    # 完全重現三階段訓練流程
    for epoch in range(total_epochs):
        if epoch < epochs_stage1:
            stage, loss_fn = 1, mse_loss
            if epoch == 0:
                optimizer = optim.Adam(model.parameters(), lr=best_params['lr1'], weight_decay=best_params['wd1'])
        elif epoch < epochs_stage1 + epochs_stage2:
            stage, loss_fn = 2, gaussian_nll_loss
            if epoch == epochs_stage1:
                for param in model.features.parameters(): param.requires_grad = False
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

    # 最終驗證與儲存
    model.eval()
    all_val_preds, all_val_labels = [], []
    with torch.no_grad():
        for maps, labels in val_loader:
            maps, labels = maps.to(device), labels.to(device)
            outputs = model(maps, apply_scaler=True)
            all_val_preds.append(outputs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    pred_mean = all_val_preds[:, [0, 2]]
    pred_log_var = all_val_preds[:, [1, 3]]
    pred_errorbar = np.sqrt(np.exp(pred_log_var))
    final_score = Score._score_phase1(true_cosmo=all_val_labels, infer_cosmo=pred_mean, errorbar=pred_errorbar)
    
    print(f"Final Model Validation Score: {final_score:.4f}")
    print(f"Final Scalers -> om: {model.log_var_scaler_om.item():.4f}, s8: {model.log_var_scaler_s8.item():.4f}")
    torch.save(model.state_dict(), 'best_model_3_stage.pth')
    print("Final model saved to best_model_3_stage.pth")

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

