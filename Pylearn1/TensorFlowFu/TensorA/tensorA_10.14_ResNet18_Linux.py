# ResNet18
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
import datetime


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def get_data(batch_num=110):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(1000).map(preprocess).batch(batch_num)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
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
    for i in range(dense_block_num - 1):
        sq2.add(DenseBlock(filter_num, stride))
    return sq2


class ResNet(tf.keras.Model):
    def __init__(self, dense_block_num, num_classes=10):
        super(ResNet, self).__init__()
        self.sq3 = Sequential([
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
        self.sq4.add(layers.Dense(512, activation=tf.nn.relu))
        self.sq4.add(layers.Dense(256, activation=tf.nn.relu))
        self.sq4.add(layers.Dense(128, activation=tf.nn.relu))
        self.sq4.add(layers.Dense(10, activation=None))

    def call(self, inputs, training=None, mask=None):
        out = self.sq3(inputs)
        out = self.sq4(out)
        return out

    def build(self, input_shape=None):
        if self._is_graph_network:
            self._init_graph_network(self.inputs, self.outputs, name=self.name)
        else:
            if input_shape is None:
                raise ValueError('You must provide an input_shape argument.')
            input_shape = tuple(input_shape)
            self._build_input_shape = input_shape
            super(ResNet, self).build(input_shape)
        self.built = True


def train_resnet(x_train, y_train, x_test, y_test, train_db, test_db):
    resnet_sq = ResNet([2, 2, 2, 2])  # resnet18
    resnet_sq.build(input_shape=[4, 32, 32, 3])
    list_loss = []
    loss_class = losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(learning_rate=0.001, name='SGD')
    # 训练模型，并且每个epoch测试一次模型精度
    for epoch1 in range(100):
        for step1, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape1:
                out = resnet_sq(x_train, training=True)
                y_train_onehot = tf.one_hot(y_train, depth=10)
                loss = loss_class(y_train_onehot, out)
                list_loss.append(loss)
            grads = tape1.gradient(loss, resnet_sq.trainable_variables)
            optimizer.apply_gradients(zip(grads, resnet_sq.trainable_variables))
            if step1 % 20 == 0:
                print(step1, end='|')
        print('\n')
        accuracy_list = []
        for step2, (x_test, y_test) in enumerate(test_db):
            predict_out = resnet_sq(x_test)
            predict_softmax = tf.nn.softmax(predict_out, axis=1)
            predict_argmax = tf.argmax(predict_softmax, axis=1)
            predict_equal = tf.equal(predict_argmax, tf.cast(y_test, dtype=tf.int64))
            predict_cnt = 0
            for i1 in predict_equal:
                if tf.equal(tf.constant(True), i1.numpy()):
                    predict_cnt += 1
            predict_result = (predict_cnt/len(predict_equal))
            if step2 % 20 == 0:
                print(step2, end='¦')
            accuracy_list.append(predict_result)
        predict_result = tf.reduce_mean(accuracy_list).numpy()
        print('\nEPOCH:', epoch1, 'LOSS:', loss.numpy(), 'Accuracy:', predict_result)
    return list_loss


def plot_tensorboard(list_loss):
    image_name = 'D:/boyLen/py/Pylearn1/image/image_tb_resnet_0'
    plt.plot(range(len(list_loss)), list_loss, color='blue')
    plt.title("ResNet18")
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.show()


if __name__ == '__main__':
    x__train, y__train, x__test, y__test, train__db, test__db = get_data()
    print('GetData:', tf.shape(x__train).numpy(), tf.shape(y__train).numpy())
    list_loss_plot = train_resnet(x__train, y__train, x__test, y__test, train__db, test__db)

