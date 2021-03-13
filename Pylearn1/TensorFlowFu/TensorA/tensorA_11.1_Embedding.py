# Embedding测试
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy


def learn1():
    # Embedding验证
    x = tf.range(10)
    x1 = tf.random.shuffle(x)
    # 创建共10个单词，每个单词用长度为4的向量表示的层
    net = layers.Embedding(10, 4)
    out = net(x1)
    print(x, x1, out)
    # 查看Embedding层内部的查询表
    print(net.embeddings)
    print(net.embeddings.trainable)
    net.trainable = False
    print(net.get_weights())
    print(net.get_config())


if __name__ == '__main__':
    learn1()