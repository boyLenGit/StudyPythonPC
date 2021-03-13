# 实现论文内容
# ***说明***
# 改模型测试了不同网络结构下的效果，证明去掉池化、上采样的loss最小；效果排名：去掉池化上采样、1234BN > 1234BN > 0BN > 不带BN的原始网络

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import tensorflow.keras.datasets as datasets
import numpy
from PIL import Image
import segyio
import matplotlib.pyplot as plt
import random

input_data_dim = 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 10024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。


def get_data_v4_segy():
    path2 = 'E:/Research/data/F3_entire.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来
        segy_data = F3_entire.trace.raw[:]  # raw[:2000]可以限制读取条数为2000条
        # 将segy的trace数据转换成为list数据
    F3_entire.close()

    # ↓ segy的列数、行数调整，以适应网络模型（行数必须是4的倍数，因为有两次2维池化和2维上采样）
    segy_data_reduce = segy_data[0:70000]
    segy_data_reduce_v2 = []
    for i in range(len(segy_data_reduce)):
        segy_data_reduce_v2.append(segy_data_reduce[i][6:])

    # ↓ Tensor化与batch化segy_data_reduce_v2
    segy_data_reduce_v2 = numpy.array(segy_data_reduce_v2)
    x_train, x_test = segy_data_reduce_v2[0:60000].astype(numpy.float32) / to_be_one \
        , segy_data_reduce_v2[-10000:].astype(numpy.float32) / to_be_one
    x_train, x_test = tf.reshape(x_train, (-1, 456)), tf.reshape(x_test, (-1, 456))
    x_train_expand_dims, x_test_expand_dims = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_test, axis=2)  # (60000, 462, 1)
    x_train_dataset, x_test_dataset = tf.data.Dataset.from_tensor_slices(x_train_expand_dims).batch(batch_size=100) \
        , tf.data.Dataset.from_tensor_slices(x_test_expand_dims).batch(batch_size=plot_show_num)
    return x_train_dataset, x_test_dataset


class One_Conv(keras.Model):
    def __init__(self):
        super(One_Conv, self).__init__()
        self.sq1 = Sequential([
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[0], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[2], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[3], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=input_data_dim, kernel_size=conv_kernel_size[4], activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


def train(db_train, db_test, train_time=100):
    modle1 = One_Conv()
    modle1.build(input_shape=(None, 456, input_data_dim))
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(train_time):
        print('---->{0}'.format(epoch1))
        for step, x_train in enumerate(db_train):
            with tf.GradientTape() as tape:
                x_result = modle1(x_train)
                # print(step, '========================x_result========================\n', tf.reshape(x_result[20], [456]), '\n', step, '========================x_train========================\n', tf.reshape(x_train[20], [456]))
                loss1 = tf.reduce_mean(tf.square(x_train - x_result))
                grads = tape.gradient(loss1, modle1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, modle1.trainable_variables))
            if step % 100 == 0: print(epoch1, step, float(loss1))


        # ↓ 测试模型结果
        x_test = []
        x_test = next(iter(db_test))
        x_test_result = modle1(x_test)
        x_test, x_test_result = tf.reshape(x_test, [-1, 456]), tf.reshape(x_test_result, [-1, 456])
        x_test, x_test_result = x_test.numpy(), x_test_result.numpy()
        x_test, x_test_result = x_test * to_be_one, x_test_result * to_be_one
        # 绘图部分
        if epoch1 == 0:
            boylen_plot(x_test, title='Input')
        if epoch1 % 2 == 0:
            boylen_plot(x_test_result, title='Result{0}'.format(epoch1))


def boylen_plot(input_list, line_num=plot_show_num, title='title'):
    list_x = []
    for i1 in range(len(input_list[0])):  # ← 生成x轴的标签数据
        list_x.append(i1)
    for i1 in range(line_num):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += 30000 * i1
    for i2 in range(line_num):
        upper, lower = 3000 + i2 * 30000, -3000 + i2 * 30000
        big_list = numpy.ma.masked_where(input_list[i2] <= upper - 500, input_list[i2])
        mid_list = numpy.ma.masked_where((input_list[i2] <= lower - 3000) | (input_list[i2] >= upper + 3000), input_list[i2])
        lit_list = numpy.ma.masked_where(input_list[i2] >= lower + 500, input_list[i2])
        plt.plot(big_list, list_x, 'b-', mid_list, list_x, 'r-', lit_list, list_x, 'b-', linewidth=1)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    db_train, db_test = get_data_v4_segy()
    #db_train, db_test = get_data_v3()
    train(db_train=db_train, db_test=db_test)
