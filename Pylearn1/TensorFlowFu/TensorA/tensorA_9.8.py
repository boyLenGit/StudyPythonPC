# 处理欠拟合与过拟合。技术点：生成分布图、动态层神经网络、自适应网络输入输出
import datetime
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, optimizers, datasets  # 导入常见网络层类

gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取所有 GPU 设备列表
if gpus:
    try:
        for gpu in gpus:    # 设置 GPU 显存占用为按需分配
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:  # 异常处理
        print(e)

# TensorBoard的路径参数声明
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape2/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# tensorboard --logdir=D:\boyLen\py\Pylearn1\logs\gradient_tape2\ --host=192.168.3.22
# 浏览器：http://192.168.3.22:6006/


def get_data_minst():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # 将图片x数据转化为张量，并缩放到(1,-1)之间
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)  # 将图片编号y转化为张量
    x_train = tf.reshape(x_train, (-1, 28 * 28))  # 改变视图， [b, 28, 28] => [b, 28*28]，即打平
    y_train = tf.one_hot(y_train, depth=10)
    return x_train, y_train, x_train, x_test, y_train, y_test


def get_data_moon():
    dot_simple_num = 2055
    rate_data_test = 0.3
    x, y = make_moons(n_samples=dot_simple_num, noise=0.3, random_state=100)  # 生成的是numpy数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate_data_test, random_state=42)
    return x, y, x_train, x_test, y_train, y_test


def draw_plot(x, y, plot_name=None, save_file_name=None):
    plt.title(plot_name)
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, cmap=plt.cm.Spectral, edgecolors='k', zorder=1)


def draw_plot_distributed(x_train, y_train, x, y, sq1, i):
    image_name = 'D:/boyLen/py/Pylearn1/image/image_tensorboard_{0}'.format(i)
    x1_max = tf.reduce_max(x_train[:, 0]) + 0.3
    x1_min = tf.reduce_min(x_train[:, 0]) - 0.3
    x2_max = tf.reduce_max(x_train[:, 1]) + 0.3
    x2_min = tf.reduce_min(x_train[:, 1]) - 0.3
    num_plot = 500j
    linspace1 = numpy.mgrid[x1_min:x1_max:num_plot, x2_min:x2_max:num_plot]  # linspace1为[2 500 500]
    flat_xy = numpy.stack((linspace1[0].flat, linspace1[1].flat), axis=1)  # 将[2 500 500]转换为per[250000,2]
    list_color = (sq1.predict_classes(flat_xy)).reshape(linspace1[0].shape)  # 可以是predict_classes或predict
    plt.pcolormesh(linspace1[0], linspace1[1], list_color)  # cmap=plt.cm.Spectral
    plot_name = "DENSE:{0}".format(2 + i)
    draw_plot(x, y, plot_name=plot_name)
    plt.savefig(image_name)
    # image_file = open(image_name+'.png', 'r')
    image_file = tf.io.read_file(image_name+'.png')
    image_tensor = tf.image.decode_png(image_file)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    with train_summary_writer.as_default():  # 调用TensorBoard
        tf.summary.image('Loss', image_tensor, step=i)


def neural_dense(x_train, y_train, x, y, epochs=20):
    len_input = tf.shape(x_train).numpy()  # 用来为下面dense计算输入规模
    len_output = tf.shape(y_train).numpy()  # 用来为下面dense计算输出规模
    len_output = len_output if len(len_output) is not 1 else [1, 1]  # 检测len_output是否是1维数据，如果是1维则标签只有1个
    print('输入规模：', len_input[1], '输出规模：', len_output[1])
    for i in range(10):
        sq1 = Sequential()
        sq1.add(layers.Dense(10*len_input[1], input_dim=len_input[1], activation='relu'))
        for ii in range(i):
            sq1.add(layers.Dense(20*len_input[1], activation='relu'))
        sq1.add(layers.Dense(len_output[1], activation='sigmoid'))
        sq1.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        sq1.fit(x_train, y_train, batch_size=2000, epochs=epochs, verbose=10)
        # 绘图部分
        draw_plot_distributed(x_train, y_train, x, y, sq1, i)


def control_center():
    x, y, x_train, x_test, y_train, y_test = get_data_moon()
    neural_dense(x_train, y_train, x, y, epochs=100)


if __name__ == '__main__':
    control_center()
