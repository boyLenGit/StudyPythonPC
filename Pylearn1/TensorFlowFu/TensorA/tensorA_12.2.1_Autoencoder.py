# 12.2 MNIST图片重建实战
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import matplotlib.pyplot as plt
import numpy
from PIL import Image

batch_num, total_vocabulary, max_sentence_len, embedding_len = 128, 10000, 80, 100


def get_data():
    # 加载Fashion MINST图片数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # 数据统一调到0-1区间，并改为float32格式
    x_train, x_test = x_train.astype(numpy.float32) / 255., x_test.astype(numpy.float32) / 255.
    # 数据切片，打乱数据，设置每次训练的batch数量
    train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(batch_size=batch_num)
    test_db = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=batch_num)
    # 因为自编码器只用到输入x，所以不用处理标签y
    return train_db, test_db


def save_images(imgs, path):  # imgs的shape：[256  28  28]，由[x, x_verify]构成
    image_all = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):  # 生成0-280的数，步幅为28，共10个循环
        for j in range(0, 280, 28):  # 生成0-280的数，步幅为28，共10个循环
            image_single = imgs[index]
            image_single = Image.fromarray(image_single, mode='L')  # numpy.array转换成image
            image_all.paste(image_single, (i, j))  # (i, j)就是像素值，用来设定 每个小图的位置；paste是拼接图像的
            index += 1
    image_all.save(path)


def save_images_compare(imgs_numpy, path):
    image_all = Image.new('L', size=(2*28, 128*28))
    index_image = 0
    for i0 in range(0, 2*28, 28):
        for i1 in range(0, 28*128, 28):
            image_single = Image.fromarray(imgs_numpy[index_image], mode='L')
            image_all.paste(image_single, (i0, i1))
            index_image += 1
    image_all.save(path)


# 自编码器模型，包含Encoder和Decoder
class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 创建encoder网络
        self.encoder_ae = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(20)
        ])
        # 创建decoder网络
        self.decoder_ae = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    # 前向传播算法
    def call(self, inputs, training=None, mask=None):
        # [b, 784] => [b, 10] => [b, 784]； 即：编码获得隐藏向量-->解码获得重建图片
        return self.decoder_ae(self.encoder_ae(inputs))  # ?training没有放进去


def train(train_db, test_db):
    model1 = AutoEncoder()
    model1.build(input_shape=(None, 784))
    model1.summary()
    optimizer1 = tf.optimizers.Adam(lr=1e-3)  # 优化器
    for epoch in range(100):  # 训练次数
        for step, x in enumerate(train_db):
            x = tf.reshape(x, [-1, 784])  # 打平数据：[b, 28, 28] => [b, 784]
            # 构件梯度记录器
            with tf.GradientTape() as tape:
                x_result = model1(x)  # 前向计算获得重建图片
                loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_result)
                # loss1 = tf.losses.binary_crossentropy(x, x_result, from_logits=True)
                loss1 = tf.reduce_mean(loss1)
            grads = tape.gradient(loss1, model1.trainable_variables)  # 自动求导
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 100 == 0: print(epoch, step, float(loss1))

            # 重建图片，从测试集抽取一张图片
            x = next(iter(test_db))  # x的shape：(128, 28, 28)
            result_test = model1(tf.reshape(x, [-1, 784]))  # 打平后输入到自编码器，获取已训练模型预测出来的图片
            x_verify = tf.sigmoid(result_test)
            x_verify = tf.reshape(x_verify, [-1, 28, 28])  # [b, 784] => [b, 28, 28]
            # 将原始图像x与自编码后的图像x_verify进行合并，最终生成的维度格式为：[x, x_verify]
            x_concat = tf.concat([x, x_verify], axis=0)  # 维度：[256  28  28] = [128  28  28], [128  28  28]
            x_concat = x_concat.numpy() * 255  # tf.Tensor转换为numpy.ndarray的写法
            x_concat = x_concat.astype(numpy.uint8)  # 将float32转换为uint8，因为绘图必须是uint8才可以
            save_images_compare(x_concat, 'cache/rec_epoch_%d.png' % epoch)  # 此x_concat的数据类型时为numpy.ndarray


if __name__ == '__main__':
    train_db, test_db = get_data()
    train(train_db, test_db)