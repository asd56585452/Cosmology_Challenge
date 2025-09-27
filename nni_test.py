# -*- coding: utf-8 -*-
"""
此腳本整合了 FAIR Universe 弱引力透鏡挑戰賽的起始套件與 Microsoft NNI 框架，
旨在使用神經網路結構搜尋 (Neural Architecture Search, NAS) 來自動尋找
一個適用於宇宙學參數推斷的最佳卷積神經網路 (CNN) 架構。

此腳本遵循最新的 NNI NAS API (nni.nas.experiment)，將模型空間、評估器和
搜尋策略完全定義在 Python 程式碼中。

--- [最終測試 - 官方教學 MNIST 版本] ---
此版本完全仿照 NNI 官方教學，使用 MNIST 資料集進行測試，並整合了所有除錯修正。
這是驗證 NNI Web UI 模式能否在您環境中運行的最終測試。
"""

# %% 0 - 匯入 & 設定
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

# --- NNI 相關匯入 ---
import nni
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
import nni.nas.strategy as strategy


# %% 1 - 適用於 MNIST 的可搜尋模型空間
class SearchableCNN_for_MNIST(ModelSpace):
    def __init__(self, nf=16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf), nn.ReLU(),
            LayerChoice([
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.Conv2d(nf, nf, kernel_size=5, padding=2),
            ], label='conv1_choice'),
            nn.BatchNorm2d(nf), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(nf, 2 * nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * nf), nn.ReLU(),
            LayerChoice([
                nn.Conv2d(2 * nf, 2 * nf, kernel_size=3, padding=1),
                nn.Conv2d(2 * nf, 2 * nf, kernel_size=5, padding=2),
            ], label='conv2_choice'),
            nn.BatchNorm2d(2 * nf), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        fc1_in_features = 2 * nf * 7 * 7
        self.hidden_features = nni.choice('hidden_features', [64, 128, 256])
        self.classifier_fc1 = MutableLinear(fc1_in_features, self.hidden_features)
        self.classifier_fc2 = MutableLinear(self.hidden_features, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.classifier_fc1(x))
        x = self.classifier_fc2(x)
        return F.log_softmax(x, dim=1)


# %% 2 - 適用於 MNIST 的模型評估器
def evaluate_model(model_cls):
    def train_epoch(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

    def test_epoch(model, device, test_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return accuracy

    model = model_cls
    N_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/mnist', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    latest_accuracy = 0.
    for epoch in range(N_EPOCHS):
        train_epoch(model, device, train_loader, optimizer, epoch + 1)
        accuracy = test_epoch(model, device, test_loader)
        latest_accuracy = accuracy
        nni.report_intermediate_result(accuracy)
        
    nni.report_final_result(latest_accuracy)


# %% 3 - 主函數：設定並啟動 NNI 實驗
def main():
    # 關鍵修正 1：延長通訊超時時間
    os.environ['NNI_PLATFORM_REST_REQUEST_TIMEOUT'] = '300s'
    root_dir = os.getcwd()
    print(f"腳本根目錄: {root_dir}")

    model_space = SearchableCNN_for_MNIST()
    evaluator = FunctionalEvaluator(evaluate_model)
    # 關鍵修正 2：使用穩定的 TPE 策略
    search_strategy = strategy.TPE()

    exp = NasExperiment(model_space, evaluator, search_strategy)
    exp.config.experiment_name = "MNIST_CNN_Search_Final_Test"
    exp.config.max_trial_number = 10
    # 關鍵修正 3：確保單 GPU 穩定運行
    exp.config.trial_concurrency = 1
    exp.config.trial_gpu_number = 1
    exp.config.max_trial_duration = '1h'
    exp.config.training_service.use_active_gpu = True

    print("正在啟動 NNI 實驗 (最終 MNIST 測試版本)...")
    exp.run(port=8089)

    print("\n搜尋結束。正在導出分數最高的模型...")
    for i, model_dict in enumerate(exp.export_top_models(top_k=3, formatter='dict')):
        print(f"Top {i+1} 模型架構: {model_dict}")
        with open(f"top_{i+1}_arch_mnist_final.json", "w") as f:
            json.dump(model_dict, f)
            
    print("\n實驗完成！您現在可以手動停止此腳本 (Ctrl+C)。")
    import time
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
