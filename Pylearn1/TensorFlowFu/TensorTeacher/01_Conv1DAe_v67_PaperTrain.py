# 实现论文内容
# ***说明***
# 改模型测试了不同网络结构下的效果，证明去掉池化、上采样的loss最小；效果排名：去掉池化上采样、1234BN > 1234BN > 0BN > 不带BN的原始网络

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy
import APIFu.LenModels.Conv1D_AE as Conv1D_AE

gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取所有 GPU 设备列表
if gpus:
    try:
        for gpu in gpus:    # 设置 GPU 显存占用为按需分配
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:  # 异常处理
        print(e)

len_kernel_size = 256
input_data_second_dim, input_data_third_dim = len_kernel_size, 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 17024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。
len_origin_single_data = 0  # 原始数据单条的长度
sample_step = 64 * 3  # 将单列数据分割成多组len_kernel_size规模数据的采样步长
path_model_weights_save = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/01_Conv1DAE/v63'
path_model_weights_load = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/01_Conv1DAE/v63_50'

path_segy_file_label = 'E:/Research/data/ArtificialData/extract_segy_valid_line_v2.segy'
path_segy_file_be_forecast = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
path_segy_file_train = 'E:/Research/data/ArtificialData/test_Mix_Noisy_v3.segy'


def get_data_v4_segy():
    segy_data = LenSegy.len_read_segy_file(path_segy_file_label)['trace_raw']
    segy_data_mix_noisy = LenSegy.len_read_segy_file(path_segy_file_train)['trace_raw']
    # 初始化数据
    global len_origin_single_data
    len_origin_single_data = len(segy_data[0])
    # ↓ segy的列数、行数调整，以适应网络模型（单条数据的长度必须是4的倍数，因为有两次2维池化和2维上采样）
    # LenSegy.len_get_segy_average_max_value(segy_data)  # 求每列的平均最大值
    # ------ test ------
    # ------
    # LenKernel处理核心
    x_, x_mix_noisy = numpy.array(boylen_kernel_generator_for_train_2input(segy_data.copy(), segy_data_mix_noisy.copy()))
    print('@ 经LenKernel后的数据规模：', x_.shape, x_mix_noisy.shape)
    # Train、Test数据范围选取
    x_train, x_train_mix_noisy = x_[:200000], x_mix_noisy[:200000]  # 这两个索引数必须一致
    x_test, x_test_mix_noisy = segy_data[-10000:], segy_data_mix_noisy[-10000:]  # 这两个索引数必须一致
    # 数据类型转化与缩放
    x_train, x_train_mix_noisy = (x_train/to_be_one).astype(numpy.float32), (x_train_mix_noisy/to_be_one).astype(numpy.float32)
    x_test, x_test_mix_noisy = (x_test/to_be_one).astype(numpy.float32), (x_test_mix_noisy/to_be_one).astype(numpy.float32)
    # 维度变形
    x_train, x_train_mix_noisy = tf.reshape(x_train, (-1, input_data_second_dim)), tf.reshape(x_train_mix_noisy, (-1, input_data_second_dim))
    x_test, x_test_mix_noisy = tf.reshape(x_test, (-1, len_origin_single_data)), tf.reshape(x_test_mix_noisy, (-1, len_origin_single_data))
    x_train, x_train_mix_noisy = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_train_mix_noisy, axis=2)
    x_test, x_test_mix_noisy = tf.expand_dims(x_test, axis=2), tf.expand_dims(x_test_mix_noisy, axis=2)
    # Batch化。x_test_all_data为全部的原始数据，x_test_all_noisy为全部加了噪声的原始数据
    x_train_dataset, x_test_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train_mix_noisy)).batch(batch_size=plot_show_num)\
        , tf.data.Dataset.from_tensor_slices((x_test, x_test_mix_noisy)).batch(batch_size=plot_show_num)
    return x_train_dataset, x_test_dataset


def boylen_kernel_generator_for_train_2input(input_list1, input_list2):  # 【核采样】指定宽度采样法（仅对x_train进行处理即可）
    enlarge_data_list1, enlarge_data_list2 = [], []
    len_input_list = len(input_list1)
    len_input_list_single = len(input_list1[0])
    print('@ LenKernel处理(train集)：       ', end='')
    for i1 in range(len_input_list):
        one_trace_cut_times = int(10 * random.random())
        step_add_all = 0
        for i2 in range(one_trace_cut_times):
            step_single = int(random.random() * (len_input_list_single-len_kernel_size-1)/3)
            step_add_all += step_single
            if (step_add_all+len_kernel_size-1) >= len(input_list1[0]): break
            enlarge_data_list1.append(input_list1[i1][step_add_all: step_add_all+len_kernel_size])
            enlarge_data_list2.append(input_list2[i1][step_add_all: step_add_all+len_kernel_size])
        if i1 % 20 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100*i1/len_input_list), '%',  end='')
    print('')
    return enlarge_data_list1, enlarge_data_list2


class One_Conv(keras.Model):
    def __init__(self):
        super(One_Conv, self).__init__()
        self.sq1 = Sequential([
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[0], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[2], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.UpSampling1D(size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[3], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.UpSampling1D(size=2),
            layers.Conv1D(filters=input_data_third_dim, kernel_size=conv_kernel_size[4], activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


def boylen_train(db_train, db_test, train_time=100):
    # 初始化数据
    global len_origin_single_data
    len_train_time = len(db_train)
    # ------------------------------------- 初始化并训练模型 ---------------------------------------
    model1 = One_Conv()
    model1.build(input_shape=(None, None, input_data_third_dim))
    model1.load_weights(path_model_weights_load)
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(51, train_time):
        print('@ 第{0}次训练：       '.format(epoch1+1), end='')
        for step, (x_train, x_train_noisy) in enumerate(db_train):
            with tf.GradientTape() as tape:
                x_result = model1(x_train_noisy)
                loss1 = tf.reduce_mean(tf.square(x_train - x_result))
                grads = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 5 == 0:
                print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_train_time), '%', end='')
        print(' ;LOSS值：', '%06f' % float(loss1), end='')
        if epoch1 % 5 == 0:
            model1.save_weights(path_model_weights_save+'_{0}'.format(epoch1))
            print('Tips:已保存模型参数', end='')
        print('')
    model1.save_weights(path_model_weights_save + '_final')


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
    datasets_train, datasets_test = get_data_v4_segy()
    boylen_train(db_train=datasets_train, db_test=datasets_test)