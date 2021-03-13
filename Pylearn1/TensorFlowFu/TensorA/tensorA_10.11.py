# 特殊神经网络学习：转置卷积；ResNet的使用
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy


# 转置卷积：o+2p-k为s倍数
def learn1():
    x = tf.range(25) + 1  # 创建 X 矩阵，高宽为 5x5
    x = tf.reshape(x, [1, 5, 5, 1])  # Reshape 为合法维度的张量
    x = tf.cast(x, tf.float32)
    w = tf.constant([[-1, 2, -3.], [4, -5, 6], [-7, 8, -9]])  # 创建固定内容的卷积核矩阵
    w = tf.expand_dims(w, axis=2)  # 将卷积核调整为合法维度的张量
    w = tf.expand_dims(w, axis=3)
    out = tf.nn.conv2d(x, w, strides=2, padding='VALID')  # 进行普通卷积运算
    # 转置卷积,使用了两种不同的API
    layers_t = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='VALID')
    # xx = tf.nn.conv2d_transpose(out, w, strides=2, padding='VALID', output_shape=[1, 6, 6, 1])
    xx = layers_t(out)
    bl = BasicBlock(1)
    bl(x)


# ResNet的使用，实现Skip Connection回退机制
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=2):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.normal1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.normal2 = layers.BatchNormalization()
        if stride != 1:  # 如果输入规模不等于输出规模，那就让输入规模变为输出规模
            self.down_sample = Sequential()
            # 这里用valid或same都一样，因为卷积核为1*1
            self.down_sample.add(layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride, padding='same'))
            # self.down_sample.add(layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride, padding='valid'))
        else:  # 如果输入规模等于输出规模，那就直接返回输入
            self.down_sample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)  # 通过第一个卷积层
        print('1:', tf.shape(out).numpy())
        out = self.normal1(out)
        out = self.relu1(out)
        out = self.conv2(out)  # 通过第二个卷积层
        print('2:', tf.shape(out).numpy())
        out = self.normal2(out)
        identity = self.down_sample(inputs)  # 输入通过 identity()转换
        print(tf.shape(identity).numpy(), tf.shape(inputs).numpy())
        output = out + identity  # f(x)+x 运算
        output = tf.nn.relu(output)  # 再通过激活函数并返回
        return output


if __name__ == '__main__':
    learn1()
