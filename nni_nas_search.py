# -*- coding: utf-8 -*-
"""
此腳本整合了 FAIR Universe 弱引力透鏡挑戰賽的起始套件與 Microsoft NNI 框架，
旨在使用神經網路結構搜尋 (Neural Architecture Search, NAS) 來自動尋找
一個適用於宇宙學參數推斷的最佳卷積神經網路 (CNN) 架構。

此腳本遵循最新的 NNI NAS API (nni.nas.experiment)，將模型空間、評估器和
搜尋策略完全定義在 Python 程式碼中。

如何執行：
1. 確認已安裝 NNI: pip install nni
2. 確認您的資料集已放置在 'public_data/' 目錄下。
3. 在終端機中執行此腳本: python nni_nas_search.py
4. 開啟終端機中顯示的 NNI WebUI 網址 (例如 http://127.0.0.1:8088) 來監控實驗進度。
"""

# %% 0 - 匯入 & 設定
import os
import json
import zipfile
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- NNI 相關匯入 ---
import nni
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear
import nni.nas.strategy as strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment


# %% 1 - 輔助類別與函數 (源自原始腳本)

# --- Utility Class ---
class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        return data + np.random.randn(*data.shape) * 0.4 / (2*ng*pixel_size**2)**0.5 * mask

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

# --- Data Class ---
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
        else: # For quick testing
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3
            self.Nsys = 20

        self.shape = [1424, 176]
        self.pixelsize_arcmin = 2
        self.ng = 30

# --- Scoring Class ---
class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        return score if score >= -10**6 else -10**6

# --- PyTorch Dataset Class ---
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

# --- Loss Function ---
def gaussian_nll_loss(output, target):
    mean_om, log_var_om = output[:, 0], output[:, 1]
    mean_s8, log_var_s8 = output[:, 2], output[:, 3]
    target_om, target_s8 = target[:, 0], target[:, 1]
    var_om = torch.exp(log_var_om)
    var_s8 = torch.exp(log_var_s8)
    loss_om = 0.5 * (log_var_om + (target_om - mean_om)**2 / var_om)
    loss_s8 = 0.5 * (log_var_s8 + (target_s8 - mean_s8)**2 / var_s8)
    return (loss_om + loss_s8).mean()

# --- GPU Noise Addition ---
def add_noise_torch(data, mask, ng, pixel_size=2.):
    noise = torch.randn_like(data) * 0.4 / (2 * ng * pixel_size**2)**0.5
    return data + noise * mask


# %% 2 - NNI：定義可搜尋的模型空間 (Model Space)
class SearchableCNN(ModelSpace):
    def __init__(self, nf=16):
        super().__init__()

        # --- 可搜尋的 Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            # LayerChoice: 讓 NNI 從數個選項中選擇一個神經層
            LayerChoice([
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.Conv2d(nf, nf, kernel_size=5, padding=2),
            ], label='conv1_choice'), # 'label' 是這個搜尋點的唯一名稱
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # --- 可搜尋的 Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(nf, 2 * nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * nf),
            nn.ReLU(),
            LayerChoice([
                nn.Conv2d(2 * nf, 2 * nf, kernel_size=3, padding=1),
                nn.Conv2d(2 * nf, 2 * nf, kernel_size=5, padding=2),
            ], label='conv2_choice'),
            nn.BatchNorm2d(2 * nf),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # --- 固定的後續 Block (為了簡化搜尋空間，也可以將它們改為可搜尋) ---
        self.final_blocks = nn.Sequential(
            nn.Conv2d(2 * nf, 4 * nf, kernel_size=3, padding=1), nn.BatchNorm2d(4 * nf), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4 * nf, 8 * nf, kernel_size=3, padding=1), nn.BatchNorm2d(8 * nf), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- 可搜尋的全連接層 ---
        # nni.choice: 讓 NNI 從數值列表中選擇一個值
        self.hidden_features = nni.choice('hidden_features', [128, 256, 512])

        self.classifier_fc1 = MutableLinear(8 * nf, self.hidden_features)
        self.classifier_fc2 = MutableLinear(self.hidden_features, 4) # 輸出4個值

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.classifier_fc1(x))
        x = self.classifier_fc2(x)
        return x


# %% 3 - NNI：定義模型評估器 (Evaluator)
def evaluate_model(model_cls):
    # NNI 會傳入一個從 SearchableCNN 空間中實例化好的模型
    model = model_cls

    # --- [修正] 將資料準備邏輯移至評估器內部 ---
    # 這樣可以避免主程序向子程序傳遞大型物件，解決序列化問題
    root_dir = os.getcwd()
    USE_PUBLIC_DATASET = True
    PUBLIC_DATA_DIR = os.path.join(root_dir, 'public_data/')
    DATA_DIR = PUBLIC_DATA_DIR if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    data_obj.mask = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.mask_file)

    # --- 訓練與評估的超參數 ---
    N_EPOCHS = 15       # 為了加速搜尋，使用較少的 Epoch
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42

    # --- 裝置與資料設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mask_tensor = torch.from_numpy(data_obj.mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- 資料分割與載入 ---
    indices = np.arange(data_obj.Nsys)
    train_indices, val_indices = train_test_split(indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    kappa_path = os.path.join(DATA_DIR, data_obj.kappa_file)
    label_path = os.path.join(DATA_DIR, data_obj.label_file)

    train_dataset = WeakLensingDataset(kappa_path, label_path, train_indices, data_obj)
    val_dataset = WeakLensingDataset(kappa_path, label_path, val_indices, data_obj)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)

    # --- 訓練與驗證迴圈 ---
    for epoch in range(N_EPOCHS):
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")
        for maps, labels in train_iterator:
            maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
            optimizer.zero_grad()
            outputs = model(maps)
            loss = gaussian_nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- 驗證與評分 ---
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]")
            for maps, labels in val_iterator:
                maps, labels = maps.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                maps = add_noise_torch(maps, mask_tensor, data_obj.ng, data_obj.pixelsize_arcmin)
                outputs = model(maps)
                all_val_preds.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

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

        # 向 NNI 報告中間結果，會顯示在 WebUI 上
        nni.report_intermediate_result(val_score)
        print(f'Epoch {epoch+1}/{N_EPOCHS}, Val Score: {val_score:.4f}')

    # 在所有 Epoch 結束後，向 NNI 報告最終結果
    nni.report_final_result(val_score)


# %% 4 - 主函數：設定並啟動 NNI 實驗
def main():
    # --- [修正] 不再需要全局變數 ---
    # global data_obj, DATA_DIR

    root_dir = os.getcwd()
    print(f"腳本根目錄: {root_dir}")

    # --- [修正] 主函數不再需要準備資料，因為這會在每個試驗中獨立完成 ---
    # # --- 1. 資料準備 ---
    # USE_PUBLIC_DATASET = True
    # PUBLIC_DATA_DIR = os.path.join(root_dir, 'public_data/')
    # DATA_DIR = PUBLIC_DATA_DIR if USE_PUBLIC_DATASET else os.path.join(root_dir, 'input_data/')
    # print(f"資料目錄: {DATA_DIR}")
    # if not os.path.exists(DATA_DIR):
    #     print("錯誤：找不到資料目錄！請確認 'public_data/' 已下載並放置在正確位置。")
    #     return
    # data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)
    # data_obj.mask = Utility.load_np(data_dir=data_obj.data_dir, file_name=data_obj.mask_file)

    # --- 2. 定義 NAS 實驗組件 ---
    model_space = SearchableCNN()
    evaluator = FunctionalEvaluator(evaluate_model)
    # 使用隨機搜尋策略，dedup=True 避免重複採樣相同的架構
    search_strategy = strategy.Random(dedup=True)

    # --- 3. 設定並啟動實驗 ---
    exp = NasExperiment(model_space, evaluator, search_strategy)

    exp.config.experiment_name = "Cosmology_CNN_Search"
    exp.config.max_trial_number = 20      # 最多嘗試 20 種不同的架構
    exp.config.trial_concurrency = 2      # 同時運行 2 個 trial (如果 GPU 資源充足)
    exp.config.trial_gpu_number = 1       # 每個 trial 使用 1 個 GPU
    
    # 如果您的 GPU 正在運行桌面環境等，必須設定此項
    exp.config.training_service.use_active_gpu = True

    # 啟動實驗，port 是 WebUI 的端口
    print("正在啟動 NNI 實驗...")
    exp.run(port=8088)

    # --- 4. 實驗結束後，導出最佳模型 ---
    print("\n搜尋結束。正在導出分數最高的模型...")
    for i, model_dict in enumerate(exp.export_top_models(top_k=3, formatter='dict')):
        print(f"Top {i+1} 模型架構: {model_dict}")
        # 將最佳架構儲存為 JSON 檔案，以供後續重新訓練使用
        with open(f"top_{i+1}_arch.json", "w") as f:
            json.dump(model_dict, f)
    print("\n實驗完成！您現在可以手動停止此腳本 (Ctrl+C)。")
    # 等待使用者手動停止，以保持 WebUI 開啟
    import time
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()

