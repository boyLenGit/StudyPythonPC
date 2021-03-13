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

len_kernel_size = 256
input_data_second_dim, input_data_third_dim = len_kernel_size, 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 20024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。


def get_data_v4_segy():
    path2 = 'E:/Research/data/F3_entire.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来,此时为ndarray格式数据（可以用numpy.array(data1)生成）
        segy_data = F3_entire.trace.raw[:]  # raw[:2000]可以限制读取条数为2000条
    F3_entire.close()

    # ↓ segy的列数、行数调整，以适应网络模型（行数必须是4的倍数，因为有两次2维池化和2维上采样）
    segy_data_reduce = segy_data[0:70000]
    segy_data_reduce_v2 = []
    for i in range(len(segy_data_reduce)):
        segy_data_reduce_v2.append(segy_data_reduce[i][6:])

    # ↓ Tensor化与batch化segy_data_reduce_v2
    segy_data_reduce_v2 = numpy.array(segy_data_reduce_v2)
    # Cut数据
    x_train, x_test = segy_data_reduce_v2[0:60000], segy_data_reduce_v2[-10000:]
    # boyLen Kernel处理核心
    x_train = numpy.array(boylen_kernel_generator(x_train.copy()))
    # 生成带噪声数据
    x_train_noisy, x_test_noisy = len_make_noisy(x_train.copy()), len_make_noisy(x_test.copy())  # ！！！此处必须加.copy()
    # 数据类型转化与缩放
    x_train, x_test = (x_train/to_be_one).astype(numpy.float32), (x_test/to_be_one).astype(numpy.float32)
    x_train_noisy, x_test_noisy = (x_train_noisy/to_be_one).astype(numpy.float32), (x_test_noisy/to_be_one).astype(numpy.float32)
    # 维度变形
    x_train, x_test = tf.reshape(x_train, (-1, input_data_second_dim)), tf.reshape(x_test, (-1, 456))
    x_train_noisy, x_test_noisy = tf.reshape(x_train_noisy, (-1, input_data_second_dim)), tf.reshape(x_test_noisy, (-1, 456))
    x_train_expand_dims, x_test_expand_dims = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_test, axis=2)  # (60000, 462, 1)
    x_train_noisy, x_test_noisy = tf.expand_dims(x_train_noisy, axis=2), tf.expand_dims(x_test_noisy, axis=2)
    # Batch化
    x_train_dataset, x_test_dataset = tf.data.Dataset.from_tensor_slices((x_train_expand_dims, x_train_noisy)).batch(batch_size=plot_show_num), tf.data.Dataset.from_tensor_slices((x_test_expand_dims, x_test_noisy)).batch(batch_size=plot_show_num)
    return x_train_dataset, x_test_dataset


def len_make_noisy(input_list):
    print('生成带噪声数据：     ', end='')
    len_input = len(input_list)
    for i1 in range(len_input):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += (random.random() * segy_max_value * 0.25)
        if i1 % 100 == 0:
            print('\b\b\b\b\b', '%02d' % (100*i1//len_input), '%',  end='')
    print('')
    return input_list  # 返回已经加噪声的数据


def boylen_kernel_generator(input_list):  # 指定宽度采样法（仅对x_train进行处理即可）
    enlarge_data_list = []
    len_input_list = len(input_list)
    len_input_list_single = len(input_list[0])
    print('boyLen核采样：     ', end='')
    for i1 in range(len_input_list):
        one_trace_cut_times = int(10 * random.random())
        step_add_all = 0
        for i2 in range(one_trace_cut_times):
            step_single = int(random.random() * (len_input_list_single-len_kernel_size-1)/3)
            step_add_all += step_single
            if (step_add_all+len_kernel_size-1) >= len(input_list[0]): break
            enlarge_data_list.append(input_list[i1][step_add_all: step_add_all+len_kernel_size])
        if i1 % 100 == 0:
            print('\b\b\b\b\b', '%02d' % (100*i1//len_input_list), '%',  end='')
    print('')
    # print('len:', len(enlarge_data_list), len(enlarge_data_list[0]), i1, i2, len(input_list[0]))
    return enlarge_data_list


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
            layers.Conv1D(filters=input_data_third_dim, kernel_size=conv_kernel_size[4], activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


def train(db_train, db_test, train_time=100):
    model1 = One_Conv()
    model1.build(input_shape=(None, input_data_second_dim, input_data_third_dim))
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(train_time):
        print('---->{0}'.format(epoch1))
        for step, (x_train, x_train_noisy) in enumerate(db_train):
            with tf.GradientTape() as tape:
                x_result = model1(x_train_noisy)
                loss1 = tf.reduce_mean(tf.square(x_train - x_result))
                grads = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 100 == 0: print(epoch1, step, float(loss1))

        # ↓ 测试模型结果
        x_test_list, x_test_result_list, x_test_noisy_list = [], [], []
        for step, (x_test, x_test_noisy) in enumerate(db_test):
            x_test_result = model1(x_test_noisy)
            x_test_list.append(x_test)
            x_test_result_list.append(x_test_result)
            x_test_noisy_list.append(x_test_noisy)
        x_test, x_test_result, x_test_noisy = tf.reshape(x_test_list, [-1, 456]), tf.reshape(x_test_result_list, [-1, 456]), tf.reshape(x_test_noisy_list, [-1, 456])
        x_test, x_test_result, x_test_noisy = x_test.numpy(), x_test_result.numpy(), x_test_noisy.numpy()
        x_test, x_test_result, x_test_noisy = x_test * to_be_one, x_test_result * to_be_one, x_test_noisy * to_be_one
        # 绘图部分
        if epoch1 == 0:
            boylen_plot(x_test, title='Input')
            boylen_plot(x_test_noisy, title='Input_noisy')
        if epoch1 % 2 == 0:
            boylen_plot(x_test_result, title='Result{0}'.format(epoch1))


def boylen_plot(input_list, line_num=plot_show_num, title='title'):
    list_x = []
    plt.figure(figsize=(12, 12))
    for i0 in range(len(input_list[0])):  # ← 生成x轴的标签数据
        list_x.append(i0)
    for i1 in range(line_num):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += 30000 * i1
    for i3 in range(line_num):
        upper, lower = 3000 + i3 * 30000, -3000 + i3 * 30000
        big_list = numpy.ma.masked_where(input_list[i3] <= upper - 500, input_list[i3])
        mid_list = numpy.ma.masked_where((input_list[i3] <= lower - 3000) | (input_list[i3] >= upper + 3000), input_list[i3])
        lit_list = numpy.ma.masked_where(input_list[i3] >= lower + 500, input_list[i3])
        plt.plot(big_list, list_x, 'b-', mid_list, list_x, 'r-', lit_list, list_x, 'b-', linewidth=2)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    db_train, db_test = get_data_v4_segy()
    train(db_train=db_train, db_test=db_test)
