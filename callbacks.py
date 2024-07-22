###<概要>コールバックのオプションを記述

import os
import tensorflow as tf
from time import gmtime, strftime 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

#早期終了
def early_stop():
    early_stopping = EarlyStopping(
        monitor='val_loss',  # モニタリングする指標。ここでは検証データの損失をモニタリングしています。
        patience=10,         # 指定したエポック数（ここでは10）改善が見られない場合、トレーニングを停止します。
        verbose=1            # 途中経過を表示します。
    )
    return early_stopping

#重み保存
def checkpoint():
    weights_dir='./weights/'
    if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
    model_checkpoint = ModelCheckpoint(
        weights_dir + "val_loss{val_loss:.3f}.hdf5",  # モデルの重みを保存するパスとファイル名の設定。val_lossは検証データの損失を示します。
        monitor='val_loss',       # モニタリングする指標。ここでは検証データの損失をモニタリングしています。
        verbose=1,                # 途中経過を表示します。
        save_best_only=True,      # 最良のモデルのみ保存します。
        save_weights_only=True,   # モデルの重みのみを保存します。
        period=3                  # エポックごとにモデルを保存する間隔を設定します。ここでは3エポックごとに保存します。
    )
    return model_checkpoint

#学習率減少
def reduce_learnrate():
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # モニタリングする指標。ここでは検証データの損失をモニタリングしています。
        factor=0.1,           # 学習率を減少させる割合。ここでは10%減少させます。
        patience=3,           # 指定したエポック数（ここでは3）改善が見られない場合、学習率を減少させます。
        verbose=1             # 途中経過を表示します。
    )
    return reduce_lr

#テンソルボード
def make_tensorboard(set_dir_name=''):
    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard