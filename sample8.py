#CNNの実装：LeNet

#ライブラリのインポート
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#GPU設定
# 利用可能な物理デバイスを取得します。
physical_devices = tf.config.experimental.list_physical_devices()

# GPUが利用可能かどうかを確認します。
gpus = [device for device in physical_devices if device.device_type == 'GPU']
if gpus:
    try:
        # 利用可能なGPUがある場合、最初のGPUを選択します（インデックス0）。
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # 必要に応じて、GPUのメモリ割り当てを制限することもできます。
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4095)])

        # TensorFlowによってGPUが使用されることを確認します。
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        
    except RuntimeError as e:
        # 明示的にGPUを設定できない場合、エラーが発生します。
        print(e)

else:
    print("GPUは利用できません。CPUでのみ計算が行われます。")
    
#モデルの定義
def lenet(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(20, kernel_size=5, padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(50, kernel_size=5, padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2,2)),
        #ここから全結合層
        Flatten(), #特徴マップをベクトル変換
        Dense(500, activation="relu"),
        Dense(num_classes),
        Activation("softmax"),
        ])
    
    model.summary() #モデル概要の出力
    return model

#データセットの準備
#データのロード
class MNISTDataset():
    def __init__(self):
        self.image_shape = (28, 28, 1) #画像のサイズ(grayscale)
        self.num_classes = 10 #クラス数

#画像データの正規化とラベルのone-hotベクトル変換
    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in [y_train, y_test]]
        
        return x_train, y_train, x_test, y_test

#データセットに加える処理を記述    
    def preprocess(self, data, label_data=False):
        if label_data:
            data = tf.keras.utils.to_categorical(data, self.num_classes) #ラベルをone-hotベクトルに変換
        
        else:
            data = data.astype("float32") #データ型の変換
            data /= 255 #画像データの値を0~1の範囲に収める
            shape = (data.shape[0],) + self.image_shape #
            print(shape)
            data = data.reshape(shape)
            
        return data

#ネットワーク学習
class Trainer():
    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logdir") #現在のディレクトリに、"log_dir"というサブディレクトリを追加したパスを生成
    
    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        #ログのクリア
        if os.path.exists(self.log_dir): #指定したパスが存在するか確認
            import shutil
            shutil.rmtree(self.log_dir) #ディレクトリの削除
        os.mkdir(self.log_dir) #ディレクトリの作成
        
        #モデルの学習
        self._target.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split, callbacks=[TensorBoard(self.log_dir)],
                        verbose=self.verbose)

#実際の処理の実装
dataset = MNISTDataset()

#モデル作成
model = lenet(dataset.image_shape, dataset.num_classes)

#モデル学習
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss = "categorical_crossentropy", optimizer=Adam())
trainer.train (x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)

#結果の表示
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])