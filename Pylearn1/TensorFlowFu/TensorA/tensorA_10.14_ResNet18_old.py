# ResNet18实战，对比CIFAR10
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy


gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取所有 GPU 设备列表
if gpus:
    try:
        for gpu in gpus:    # 设置 GPU 显存占用为按需分配
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:  # 异常处理
        print(e)


def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def get_data(batch_num=100):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()  # 在线下载，加载 CIFAR10 数据集
    y_train = tf.squeeze(y_train, axis=1)  # 删除 y 的一个维度，[b,1] => [b]
    y_test = tf.squeeze(y_test, axis=1)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 构建训练集对象
    train_db = train_db.shuffle(1000).map(preprocess).batch(batch_num)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集对象
    test_db = test_db.map(preprocess).batch(batch_num)
    x_train = tf.cast(x_train, dtype=tf.float32)
    return x_train, y_train, x_test, y_test, train_db, test_db


class DenseBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(DenseBlock, self).__init__()
        self.sq1 = Sequential()
        self.sq1.add(layers.Conv2D(filter_num, strides=stride, kernel_size=(3, 3), padding='same'))
        self.sq1.add(layers.BatchNormalization())
        self.sq1.add(layers.ReLU())
        self.sq1.add(layers.Conv2D(filter_num, strides=1, kernel_size=(3, 3), padding='same'))
        self.sq1.add(layers.BatchNormalization())
        if stride != 1:
            self.down_sample = Sequential()
            self.down_sample.add(layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride, padding='same'))
        else:
            self.down_sample = lambda x: x

    def call(self, inputs, training=None):
        out = self.sq1(inputs)
        skip_x = self.down_sample(inputs)
        out = tf.nn.relu(out + skip_x)
        return out


def add_dense_block(filter_num, dense_block_num, stride=1):
    sq2 = Sequential()
    sq2.add(DenseBlock(filter_num, stride))
    for i in range(dense_block_num-1):
        sq2.add(DenseBlock(filter_num, stride))
    return sq2


class ResNet(tf.keras.Model):
    def __init__(self, dense_block_num, num_classes=10):
        super(ResNet, self).__init__()
        self.sq3 = Sequential([  # 根网络，数据处理
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])
        self.sq4 = Sequential()
        self.sq4.add(add_dense_block(64, dense_block_num[0], stride=1))
        self.sq4.add(add_dense_block(64, dense_block_num[1], stride=2))
        self.sq4.add(add_dense_block(64, dense_block_num[2], stride=2))
        self.sq4.add(add_dense_block(64, dense_block_num[3], stride=2))
        self.sq4.add(layers.GlobalAveragePooling2D())
        self.sq4.add(layers.Flatten())
        self.sq4.add(layers.Dense(256, activation=tf.nn.relu))
        self.sq4.add(layers.Dense(128, activation=tf.nn.relu))
        self.sq4.add(layers.Dense(10, activation=None))

    def call(self, inputs, training=None, mask=None):
        out = self.sq3(inputs)
        out = self.sq4(out)
        return out

    def build(self, input_shape=None):  # 如果不复制build函数，则会报错，看来继承Model必须继承init、call、build才可以
        if self._is_graph_network:
            self._init_graph_network(self.inputs, self.outputs, name=self.name)
        else:
            if input_shape is None:
                raise ValueError('You must provide an `input_shape` argument.')
            input_shape = tuple(input_shape)
            self._build_input_shape = input_shape
            super(ResNet, self).build(input_shape)
        self.built = True


def train_resnet(x_train, y_train, x_test, y_test, train_db, test_db):
    resnet_sq = ResNet([2, 2, 2, 2])  # resnet18
    resnet_sq.build(input_shape=[4, 32, 32, 3])
    list_loss = []
    # resnet_sq.compile(optimizer=optimizers.Adam(lr=0.01), loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # resnet_sq.fit(x_train, y_train, batch_size=30, epochs=10)  # 规模：[50000    32    32     3] [50000     1]
    for epoch1 in range(100):
        for step1, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape1:
                out = resnet_sq(x_train)
                y_train_onehot = tf.one_hot(y_train, depth=10)
                loss = tf.losses.categorical_crossentropy(y_train_onehot, out, from_logits=True)
                loss = tf.reduce_mean(loss)
                list_loss.append(loss)
            grads = tape1.gradient(loss, resnet_sq.trainable_variables)
            optimizers.Adam(lr=1e-4).apply_gradients(zip(grads, resnet_sq.trainable_variables))
            if step1 % 10 == 0:
                print(step1, end='|')
        print('/n', epoch1, 'LOSS:', loss.numpy())
    return list_loss


if __name__ == '__main__':
    x__train, y__train, x__test, y__test, train__db, test__db = get_data()
    print('GetData数据规模：', tf.shape(x__train).numpy(), tf.shape(y__train).numpy())
    list_loss_plot = train_resnet(x__train, y__train, x__test, y__test, train__db, test__db)
