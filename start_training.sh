#!/bin/bash

# 進入您的專案目錄
cd /home/ubuntu/Cosmology_Challenge/

# 啟動您的 Conda 虛擬環境
# 注意：您可能需要提供 conda.sh 的完整路徑
source /opt/pytorch/bin/activate

# 執行您的 Python 訓練腳本
# 加上 --resume 參數，讓程式從最新的檢查點恢復
echo "Starting training at $(date)" > training.log
python Phase_1_Startingkit_WL_PSAnalysis.py >> training.log 2>&1

# cd Cosmology_Challenge
# aws s3 cp s3://cosmologychallengebucket/public_data.zip .
# unzip public_data.zip -d public_data
# rm public_data.zip
# chmod +x /home/ubuntu/Cosmology_Challenge/start_training.sh
# crontab -e
# @reboot /usr/bin/tmux new-session -d -s Cosmology_Session '/home/ubuntu/Cosmology_Challenge/start_training.sh'
# restart
