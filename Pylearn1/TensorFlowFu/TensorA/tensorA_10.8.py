# 卷积神经网络练习
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy


def learn1():
    # 构造输入
    x = tf.random.normal([100, 32, 32, 3])
    # 将其他维度合并，仅保留通道维度
    x = tf.reshape(x, [-1, 3])
    # 计算其他维度的均值
    ub = tf.reduce_mean(x, axis=0)
    print(ub)


if __name__ == '__main__':
    learn1()