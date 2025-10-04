#!/bin/bash

# 進入您的專案目錄
cd /home/ubuntu/Cosmology_Challenge/

# 啟動您的 Conda 虛擬環境
# 注意：您可能需要提供 conda.sh 的完整路徑
source /opt/pytorch/bin/activate

# 執行您的 Python 訓練腳本
# 加上 --resume 參數，讓程式從最新的檢查點恢復
python Phase_1_Startingkit_WL_PSAnalysis.py
