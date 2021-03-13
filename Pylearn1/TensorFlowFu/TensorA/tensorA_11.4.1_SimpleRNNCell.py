# SimpleRNNCell层
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Embedding层功能验证
input1 = tf.random.shuffle(tf.range(10))  # shuffle将数据随机打乱
# 创建共10个单词，每个单词用长度为4的向量表示的层。Embedding生成单词向量，可学习。
net_Embedding = layers.Embedding(input_dim=10, output_dim=4)
out = net_Embedding(input1)
print('Embedding层：', out[1])
print(net_Embedding.embeddings[0], net_Embedding.embeddings.trainable)
print('trainable:', net_Embedding.trainable)
net_Embedding.trainable = False

# SimpleRNNCell测试
net_cell = layers.SimpleRNNCell(units=3)
net_cell.build(input_shape=(None, 4))
print('net_cell.trainable_variables:', net_cell.trainable_variables[0][0])

# 初始化状态向量
h0 = [tf.zeros([4, 64])]
x_0 = tf.random.normal([4, 80, 100])
x_1 = x_0[:, 0, :]  # 这里只要保证去掉只有1列的维后总维度只有2，那就可以计算。也就是[4,1,1,1,1,8]计算时也是二维的[4,8]。其他网络也如此
net_cell2 = layers.SimpleRNNCell(64)
out1, h1 = net_cell2(x_1, h0)
print('shape:', tf.shape(out1), tf.shape(h1), id(h1))

# 构建多层SimpleRNNCell
x = tf.random.normal([4, 80, 100])
xt = x[:, 0, :]
cell0 = layers.SimpleRNNCell(64)  # 设定输出空间的维度为64
cell1 = layers.SimpleRNNCell(64)
h0 = [tf.zeros([4, 64])]  # cell0的初始状态向量
h1 = [tf.zeros([4, 64])]
middle_sequense = []
for i1 in tf.unstack(x, axis=1):  # 在时间轴上经过多次循环来实现整个时间戳的前向计算
    out0, h0 = cell0(i1, h0)
    middle_sequense.append(out0)  # 保存第一层的所有时间戳的输出
    out1, h1 = cell1(out0, h1)  # 保存第一层中


for i3 in middle_sequense:
    out1, h1 = cell1(i3, h1)