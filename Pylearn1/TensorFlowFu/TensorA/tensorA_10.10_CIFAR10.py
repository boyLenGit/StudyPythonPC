# 卷积神经网络CIFAR10实战
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy


tf.random.set_seed(2345)
batch_num = 150


def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()  # 在线下载，加载 CIFAR10 数据集
    y_train = tf.squeeze(y_train, axis=1)  # 删除 y 的一个维度，[b,1] => [b]
    y_test = tf.squeeze(y_test, axis=1)
    print('shape:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # 打印训练集和测试集的形状
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 构建训练集对象
    train_db = train_db.shuffle(1000).map(preprocess).batch(batch_num)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集对象
    test_db = test_db.map(preprocess).batch(batch_num)
    sample = next(iter(train_db))  # 从训练集中采样一个 Batch，并观察
    print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
    return x_train, y_train, x_test, y_test, train_db, test_db
    # 维度：x_train-->(50000, 32, 32, 3)


def train_neural(train_db):  # 双容器
    sq_conv = Sequential([
        # Conv-Conv-Pooling 单元 1
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 64 个 3x3 卷积核, 输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 高宽减半
        # Conv-Conv-Pooling 单元 2,输出通道提升至 128，高宽大小减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
    ])
    sq_fc = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])
    sq_conv.build(input_shape=[None, 32, 32, 3])
    sq_fc.build(input_shape=[None, 512])
    sq_conv.summary(), sq_fc.summary()
    variables = sq_conv.trainable_variables + sq_fc.trainable_variables  # [1, 2] + [3, 4] => [1, 2, 3, 4]
    for epoch1 in range(50):
        for step1, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape1:
                out_conv = sq_conv(x_train)
                out_conv = tf.reshape(out_conv, shape=[batch_num, 512])  # 先将数据打平
                out_fc = sq_fc(out_conv)
                # out_fc = tf.nn.softmax(out_fc, axis=1)
                y_train_onehot = tf.one_hot(y_train, depth=10)
                loss = tf.losses.categorical_crossentropy(y_train_onehot, out_fc, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape1.gradient(loss, variables)
            optimizers.Adam(lr=1e-4).apply_gradients(zip(grads, variables))
        print(epoch1, 'LOSS:', loss.numpy())


def train_neural_v2(train_db):
    sq_conv = Sequential([
        # 第1层
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 64 个 3x3 卷积核, 输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),  # 高宽减半
        # 第2层,输出通道提升至 128，高宽大小减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # 第3层,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # 第4层,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # 第5层,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # 全连接层
        layers.Flatten(),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])
    sq_conv.build(input_shape=[4, 32, 32, 3])
    sq_conv.summary()
    for epoch1 in range(50):
        for step1, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape1:
                out = sq_conv(x_train)
                y_train_onehot = tf.one_hot(y_train, depth=10)
                # categorical_crossentropy是交叉熵损失函数
                loss = tf.losses.categorical_crossentropy(y_train_onehot, out, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape1.gradient(loss, sq_conv.trainable_variables)
            optimizers.Adam(lr=1e-4).apply_gradients(zip(grads, sq_conv.trainable_variables))
            if step1 % 10 == 0: print(step1, end='|')
        print('/n', epoch1, 'LOSS:', loss.numpy())


def control_center():
    x_train, y_train, x_test, y_test, train_db, test_db = get_data()
    train_neural_v2(train_db)


if __name__ == '__main__':
    control_center()
