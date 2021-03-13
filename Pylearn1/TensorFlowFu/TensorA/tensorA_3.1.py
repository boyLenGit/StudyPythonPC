import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets  # 导入tensorflow库、子库


def TrainMNIST():
    (x, y), (x_val,
             y_val) = datasets.mnist.load_data()  # 调用Tensorflow的模块，自动下载MNIST数据。下载的MINST数据会自动返回两个元组对象，前两个是训练集，后两个是测试集.x都是图片数据集，y都是对应图片的编号
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1  # 将图片x数据转化为张量，并缩放到(1,-1)之间
    y = tf.convert_to_tensor(y, dtype=tf.int32)  # 将图片编号y转化为张量
    print('y张量：', x)
    y = tf.one_hot(y, depth=10)  # one-hot编码
    print('y:', y)
    print('x.shape:', x.shape, 'y.shape:', y.shape)
    train_dataset1 = tf.data.Dataset.from_tensor_slices((x, y))  # 构件数据集对象
    print('---', train_dataset1)
    train_dataset1 = train_dataset1.batch(200)  # 将数据 batch化
    print('---', train_dataset1)
    for step, (x, y) in enumerate(train_dataset1):
        print('===', step, (x, y))


if __name__ == '__main__':
    TrainMNIST()
