# 底层BP传播，基于Tensorflow
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.datasets as datasets
import tensorflow.keras as keras


def get_data():
    dot_simple_num = 2055
    rate_data_test = 0.3
    x, y = make_moons(n_samples=dot_simple_num, noise=0.3, random_state=100)  # 生成的是numpy数据集
    # 如果不带marker=会报错，如果是marker='ro'也会报错
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, cmap=plt.cm.Spectral, edgecolors='none')
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate_data_test, random_state=42)
    return x, y, x_train, x_test, tf.one_hot(y_train, depth=2), tf.one_hot(y_test, depth=2)


def get_data_minst():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # 自动下载MNIST数据并返回两个numpy.array对象，前两个是训练集，后两个是测试集.x都是图片数据集，y都是对应图片的编号。下载的数据为numpy格式的文件
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # 将图片x数据转化为张量，并缩放到(1,-1)之间
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)  # 将图片编号y转化为张量
    x_train = tf.reshape(x_train, (-1, 28 * 28))  # 改变视图， [b, 28, 28] => [b, 28*28]，即打平
    # 原本y是y.shape: (60000, 10)，经过batch(200)后变为y.shape: (200, 10)；x则由(60000, 28, 28)先经reshape变为(60000, 784)再经batch变为(200, 784)
    return None, None, x_train, x_test, tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=2)


class NeuralNetwork:
    def __init__(self):
        self.list_layers = []

    def add_neural_layer(self, num_x, num_a_neural, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else tf.random.truncated_normal([num_x, num_a_neural], stddev=0.1)
        self.bias = bias if bias is not None else tf.random.truncated_normal([num_a_neural], stddev=0.1)  # 偏置变量
        self.activation = activation  # 激活函数标识符
        self.out_single_dense = None  # 单层的网络的最终输出，作为下一层的输入
        self.out_all_batch = None  # 全部batch的最终层d 输出y_out
        self.out_activited = None  # 经过激活函数后的输出
        self.error_fp = None  # 运算辅助变量
        self.delta_fp = None  # LOSS对w的整体梯度值
        self.delta_little = None  # LOSS对w的整体梯度值的一小部分，辅助运算
        self.layer = {'weights': self.weights, 'bias': self.bias, 'activation': self.activation,
                      'out_all_batch': self.out_all_batch,
                      'out_single_dense': self.out_single_dense, 'delta_little': self.delta_little,
                      'out_activited': self.out_activited, 'error_fp': self.error_fp, 'delta_fp': self.delta_fp}
        self.list_layers.append(self.layer)

    def forward_propagation(self, x):
        x = tf.constant(x, dtype=tf.float32)
        i = 0
        for layer in self.list_layers:
            if layer == self.list_layers[0]:  # 如果是第一层网络，则输入为x_train
                layer['out_single_dense'] = self.func_activation(x @ layer['weights'] + layer['bias'])
            else:  # 如果不是第一层网络，则输入为前一层的输出
                layer_previous = self.list_layers[i - 1]
                layer['out_single_dense'] = self.func_activation(layer_previous['out_single_dense'] @ layer['weights'] + layer['bias'])
            i += 1
        layer = self.list_layers[-1]
        layer['out_all_batch'] = layer['out_single_dense'] if layer['out_all_batch'] is None \
            else tf.concat([layer['out_all_batch'], layer['out_single_dense']], axis=0)
        # print('前向传播  i:', i, len(self.list_layers), '  out输出：', tf.shape(layer['out_single_dense']), tf.shape(x), tf.shape(layer['weights']))

    def back_propagation(self, x, y_tips, rate_learning):   # 反向传播算法
        y_tips = tf.constant(y_tips, dtype=tf.float32)  # 为了后面可以实现y_tips-y_out，因为y_out是tensorflow格式的数据
        x = tf.constant(x, dtype=tf.float32)
        for i in reversed(range(len(self.list_layers))):  # 计算全部层的梯度值
            layer = self.list_layers[i]
            if layer == self.list_layers[-1]:  # 如果是最后一层
                layer_previous = self.list_layers[i - 1]
                # print('1激活偏导数:', tf.shape(self.func_activation_derivative(layer['out_single_dense'])).numpy(), '准确标签:', tf.shape(y_tips).numpy(), '猜测标签:', tf.shape(layer['out_single_dense']).numpy(), '前一个输出:', tf.shape(layer_previous['out_single_dense']).numpy(), '当前层权重：', tf.shape(layer['weights']).numpy())
                layer['delta_little'] = self.func_activation_derivative(layer['out_single_dense']) * (y_tips - tf.nn.softmax(layer['out_single_dense']))
                layer['delta_fp'] = tf.transpose(layer['delta_little'], perm=[1, 0]) @ layer_previous['out_single_dense']
            else:  # 如果不是最后一层
                layer_last = self.list_layers[i + 1]
                if i == 0:  # 如果是第一层
                    # print('3激活偏偏导:', tf.shape(self.func_activation_derivative(layer['out_single_dense'])).numpy(), '后一层小偏导:', tf.shape(layer_last['delta_little']).numpy(), '后一层权重:', tf.shape(layer_last['weights']).numpy(), '输如x:', tf.shape(x).numpy())
                    layer['delta_little'] = self.func_activation_derivative(layer['out_single_dense']) * (layer_last['delta_little'] @ tf.transpose(layer_last['weights'], perm=[1, 0]))
                    layer['delta_fp'] = tf.transpose(layer['delta_little'], perm=[1, 0]) @ x
                else:   # 如果是中间层
                    layer_previous = self.list_layers[i - 1]
                    # print('2激活偏偏导:', tf.shape(self.func_activation_derivative(layer['out_single_dense'])).numpy(), '后一层小偏导:', tf.shape(layer_last['delta_little']).numpy(), '后一层权重:', tf.shape(layer_last['weights']).numpy(), '输如x:', tf.shape(x).numpy())
                    layer['delta_little'] = self.func_activation_derivative(layer['out_single_dense']) * (layer_last['delta_little'] @ tf.transpose(layer_last['weights'], perm=[1, 0]))
                    layer['delta_fp'] = tf.transpose(layer['delta_little'], perm=[1, 0]) @ layer_previous['out_single_dense']
        for i in range(len(self.list_layers)):  # 更新梯度到w和b中（未推导b）
            layer = self.list_layers[i]
            layer['weights'] += rate_learning * tf.transpose(layer['delta_fp'], perm=[1, 0])

    def train_neural(self, x_train, y_train, train_epochs, train_batch=1000, rate_learn=0.0075664):
        list_mse = []
        for i_batch in range(train_epochs):
            layer = self.list_layers[-1]
            layer['out_all_batch'] = None  # 初始化out_all_batch
            cnt_for_batch = len(x_train) // train_batch     # 得到总batch数，为了设置最后batch时采取的操作
            for i_x_len in range(len(x_train)):  # batch分批训练环节：FP+BP
                if (i_x_len + 1) % train_batch == 0:
                    # print(i_x_len, '-', end='')
                    self.forward_propagation(x_train[i_x_len + 1 - train_batch: i_x_len + 1])
                    self.back_propagation(x_train[i_x_len + 1 - train_batch: i_x_len + 1],
                                          y_train[i_x_len + 1 - train_batch: i_x_len + 1], rate_learn)
                    cnt_for_batch -= 1
                if cnt_for_batch == 0:  # 如果是最后一组batch
                    self.forward_propagation(x_train[i_x_len + 1: len(x_train) + 1])
                    self.back_propagation(x_train[i_x_len + 1: len(x_train) + 1], y_train[i_x_len + 1: len(x_train) + 1], rate_learn)
                    break
            layer_last = self.list_layers[-1]
            mse_single = numpy.mean(numpy.square((y_train - layer_last['out_all_batch']).numpy()))
            list_mse.append(mse_single)
            # mse_single = tf.losses.MSE(y_train, layer_last['out_all_batch'])
            # print('MSE_single:', mse_single, '---', y_train, '---', layer_last['out_all_batch'])
            print('MSE_single:', mse_single)
        return list_mse

    def train_neural_smart_dense(self):
        pass

    def func_activation(self, r):  # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数，直接返回
        if self.activation == 'relu':  # ReLU 激活函数
            return numpy.maximum(r, 0)
        elif self.activation == 'tanh':  # tanh 激活函数
            return numpy.tanh(r)
        elif self.activation == 'sigmoid':  # sigmoid 激活函数
            return 1 / (1 + numpy.exp(-r))
        return r

    def func_activation_derivative(self, r):  # 针对于不同类型的激活函数，它们的导数计算实现如下：
        if self.activation is None:  # 无激活函数，导数为1
            return numpy.ones_like(r)
        if self.activation == 'relu':  # ReLU 函数的导数实现
            grad = numpy.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':  # tanh 函数的导数实现
            return 1 - r ** 2
        elif self.activation == 'sigmoid':  # Sigmoid 函数的导数实现
            return r * (1 - r)
        return r


def control_center(train_epochs=2000):
    x, y, x_train, x_test, y_train, y_test = get_data()
    dense = NeuralNetwork()
    dense.add_neural_layer(2, 25, 'sigmoid')
    dense.add_neural_layer(25, 50, 'sigmoid')
    dense.add_neural_layer(50, 25, 'sigmoid')
    dense.add_neural_layer(25, 2, 'sigmoid')
    list_mse = dense.train_neural(x_train, y_train, train_epochs)  # 开始训练
    plt.plot(range(len(list_mse)), list_mse)
    plt.show()


def control_center_minst(train_epochs=200):
    x, y, x_train, x_test, y_train, y_test = get_data_minst()
    dense = NeuralNetwork()
    dense.add_neural_layer(784, 512, 'sigmoid')
    dense.add_neural_layer(512, 256, 'sigmoid')
    dense.add_neural_layer(256, 128, 'sigmoid')
    dense.add_neural_layer(128, 10, 'sigmoid')
    # dense.forward_propagation(x_train)
    # dense.back_propagation(x_train, y_train, 0.01)
    list_mse = dense.train_neural(x_train, y_train, train_epochs)
    plt.plot(range(len(list_mse)), list_mse)
    plt.show()


if __name__ == '__main__':
    control_center()
