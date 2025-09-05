你的目標是為弱重力透鏡挑戰賽實作、訓練和評估一個基於 PyTorch 的基準模型。嚴格遵循以下步驟：

任務目標：

設定環境：安裝 PyTorch 及其他必要的函式庫。

處理數據：建立一個 Python 腳本，該腳本需複製 Phase_1_Startingkit_WL_PSAnalysis.ipynb 筆記本中的邏輯來載入、預處理及準備數據，包括重塑數據、添加噪聲、處理標籤，並建立 PyTorch DataLoader。

實作模型：根據提供的 Keras 範例，在 PyTorch 中定義一個深度 CNN 模型。此模型應輸出四個值：Ωm 和 S8 的預測平均值及其對應的不確定性 (σ)。參考以下tensorflow的實現:

import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, Input, Dense, Activation
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2


def mycnn(imsize, nf=32, reg = 5e-5, padding='valid'):
    """Return a 512 pixel CNN."""
    # input
    inp = Input(shape=(imsize, imsize, 1))

    # conv block
    x = Conv2D(nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(inp)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(2*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(2*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(4*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(4*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(8*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = AveragePooling2D(strides=(2,2))(x)
    
    # conv block
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D( 8*nf, (1, 1), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(16*nf, (3, 3), padding=padding, kernel_regularizer=l2(reg))(x)
    x = Activation('relu')(x)
    
    # final regression
    x = GlobalAveragePooling2D()(x)
    x = Dense(2)(x)
        
    model = Model(inputs=inp, outputs=x)
    return model

訓練與評估：開發一個訓練腳本，使用競賽指定的自定義 KL 散度損失函數，在預處理後的數據上訓練模型。優化器需包含 L2 正規化 (weight_decay)，並在驗證集上評估模型性能。

產生預測：使用訓練好的模型對測試集產生預測，並將結果以競賽提交所需的 result.json 格式儲存。

打包與檢查：產生一個 zip 壓縮檔，並參考 Phase_1_Startingkit_WL_PSAnalysis.ipynb 中生成的範例輸出 submissions/Submission_25-09-05-19-21.zip，以確保最終輸出格式正確無誤。

