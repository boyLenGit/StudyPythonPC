# BP反向传播算法实战，纯底层计算梯度
# import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def AdjustData():
    dot_simple_num = 2000
    testdata_rate = 0.3
    x, y = make_moons(n_samples=dot_simple_num, noise=0.2, random_state=100)  # 生成的是numpy数据集
    # 如果不带marker=会报错，如果是marker='ro'也会报错
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, cmap=plt.cm.Spectral, edgecolors='none')
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testdata_rate, random_state=42)
    return x, y, x_train, x_test, y_train, y_test


class Layer_Active:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param int n_input: 输入节点数
        :param int n_neurons: 输出节点数
        :param str activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        # 通过正态分布初始化网络权值，初始化非常重要，不合适的初始化将导致网络不收敛
        self.weights = weights if weights is not None \
            else numpy.random.randn(n_input, n_neurons) * numpy.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else numpy.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型，如’sigmoid’
        self.last_activation = None  # 激活函数的输出值o
        self.error = None  # 用于计算当前层的delta变量的中间变量
        self.delta = None  # 记录当前层的delta 变量，用于计算梯度

    # 网络层的前向传播函数实现如下，其中last_activation 变量用于保存当前层的输出值：
    def activate(self, x):  # x是一维2数据的numpy:[,]
        r = numpy.dot(x, self.weights) + self.bias  # X@W+b；这里只有weights是二维的，其他都是一维：(25,)
        self.last_activation = self.func_activation(r)  # r通过激活函数，得到全连接层的输出o
        return self.last_activation  # self.last_activation的规模跟每层网络数有关

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

    # 针对于不同类型的激活函数，它们的导数计算实现如下：
    def func_activation_derivative(self, r):

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


class Nerual_Network:
    def __init__(self):
        self.list_layers = []

    def add_layer(self, layer):  # 添加层到list中
        self.list_layers.append(layer)  # list_layers中存储的是class Layer_Active，每层网络存储一个class Layer_Active

    def FP(self, x):  # 前向传播计算部分
        for layer in self.list_layers:  # layer是类Class对象Layer_Active
            x = layer.activate(x)
        return x  # 这里的x为一维2项数组[,]

    def BP(self, x, y, learning_rate):  # 反向传播，计算梯度并更新w
        out_FP = self.FP(x)  # out_FP是经过前向传播中的全部网络层，最终输出2节点的数值：[,]
        for i in reversed(range(len(self.list_layers))):  # 从最大的序号开始向最小的序号遍历：3-2-1-0层这样
            layer = self.list_layers[i]  # 得到当前层
            if layer == self.list_layers[-1]:  # 第-1即为list的最后一项，也就是最后一层网络，反向传播是从最后一层开始的
                layer.error = y - out_FP
                layer.delta = layer.error * layer.func_activation_derivative(layer.last_activation)  # 最末层的layer.last_activation就是out_FP
            else:
                next_layer = self.list_layers[i + 1]  # 后一层，因为是反向传播，所以是从后到前
                layer.error = numpy.dot(next_layer.weights, next_layer.delta)
                # 这里的layer.last_activation就是当前层的输出，每次循环时层会逐级递减：3-2-1-0
                layer.delta = layer.error * layer.func_activation_derivative(layer.last_activation)
        for i in range(len(self.list_layers)):  # 更新梯度部分
            layer = self.list_layers[i]
            x = (x if i == 0 else self.list_layers[i - 1].last_activation)  # 如果是第一层，那out_before就是第一层的输入x；如果不是第一层，那输入就是上一层的输出
            out_before = numpy.atleast_2d(x)
            # 因为下面用的计算符是+=，所以计算出来的结果的结构要和weights的一样，才可以相加。因此要用上面的atleast_2d与.T 来转到相同的维度。
            # weights是二维矩阵，因为weights还要变换输入-输出之间的规模。
            delta_loss = layer.delta * out_before.T  # 最终的梯度表达式 这里是矩阵相乘
            layer.weights += delta_loss * learning_rate    # .T是转置

    def Accuracy(self, x, y):
        out_FP = self.FP(x)  # 将测试集的某数据经过前向传播计算结果
        return numpy.sum(numpy.equal(numpy.argmax(out_FP, axis=1), y)) / y.shape[0]

    def Train(self, x_train, x_test, y_train, y_test, learning_rate, max_epochs):
        # 生成独热编码
        y_onehot = numpy.zeros((y_train.shape[0], 2))  # y_train.shape[0]是将(1400,)转化成1400，即为zeros(1400, 2),2是因为独热编码就2项:[0,1]
        y_onehot[numpy.arange(y_train.shape[0]), y_train] = 1  # [,]中的“,”两边都是list才可以实现逐项遍历，如果其中一边是:则不会逐项遍历，直接全为1
        list_MSE = []
        list_Accuracys = []
        for i in range(max_epochs):  # 这里+1的原因是range(10)生成的是0-9的数据，不包括10
            for j in range(len(x_train)):
                self.BP(x_train[j], y_onehot[j], learning_rate)
            y_out = self.FP(x_train)  # 前向传播计算out输出值，因为共2节点，因此输出值2项一组
            mse = numpy.mean(numpy.square(y_onehot - y_out))
            list_MSE.append(mse)
            accruacys = self.Accuracy(x_test, y_test)  # 计算模型对测试集的预测准确度
            list_Accuracys.append(accruacys)
            if i % 10 == 0:
                print('Epochs:', i, ' MSE:', mse, ' Accuracy:', accruacys)
        return list_MSE, list_Accuracys


def ControlCenter():
    x, y, x_train, x_test, y_train, y_test = AdjustData()
    neural = Nerual_Network()
    neural.add_layer(Layer_Active(2, 25, 'sigmoid'))  # 隐藏层 1, 2=>25
    neural.add_layer(Layer_Active(25, 50, 'sigmoid'))  # 隐藏层 2, 25=>50
    neural.add_layer(Layer_Active(50, 25, 'sigmoid'))  # 隐藏层 3, 50=>25
    neural.add_layer(Layer_Active(25, 2, 'sigmoid'))  # 输出层, 25=>2
    list_MSE, list_Accuracys = neural.Train(x_train, x_test, y_train, y_test, 0.01, 100)
    x_axis = [i for i in range(0, len(list_MSE))]
    # 绘制MES曲线
    plt.title("MES Loss")
    plt.plot(x_axis, list_MSE, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    # 绘制Accuracy曲线
    plt.title("Accuracy")
    plt.plot(x_axis, list_Accuracys, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    ControlCenter()
