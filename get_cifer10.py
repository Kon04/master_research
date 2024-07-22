###<概要>cifer10のデータセットを取得し、学習用の処理を加える

import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def get_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 

    #データ抽出（今回は訓練データ5000枚、テストデータ1000枚）
    X_train = X_train[:5000]
    X_test = X_test[:1000]

    #画像サイズの変更
    input_size = 139
    num=len(X_train)
    zeros = np.zeros((num,input_size,input_size,3))
    for i, img in enumerate(X_train):
        zeros[i] = cv2.resize(
            img,
            dsize = (input_size,input_size)
        )
    X_train = zeros
    del zeros

    num=len(X_test)
    zeros = np.zeros((num,input_size,input_size,3))
    for i, img in enumerate(X_test):
        zeros[i] = cv2.resize(
            img,
            dsize = (input_size,input_size)
        )
    X_test = zeros
    del zeros

    # データ型の変換＆正規化
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # one-hot変換
    num_classes = 10 
    y_train = to_categorical(y_train, num_classes = num_classes)[:5000]
    y_test = to_categorical(y_test, num_classes = num_classes)[:1000]
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test