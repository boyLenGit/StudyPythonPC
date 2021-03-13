# 实现论文内容
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import tensorflow.keras.datasets as datasets
import numpy
from PIL import Image

input_data_dim = 1
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_mnist


def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255., tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_train = tf.reshape(x_train, (-1, 28 * 28))  # 数据规模(?, 784)
    x_train = tf.expand_dims(x_train, axis=2)
    # x_train = tf.stack([x_train, x_train], axis=2)  # 构建图中的2个输入维度的输入
    y_train = tf.one_hot(y_train, depth=10)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=100)
    return db_train


def get_data_v2():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(numpy.float32) / 255., x_test.astype(numpy.float32) / 255.
    x_train, x_test = tf.reshape(x_train, (-1, 28 * 28)), tf.reshape(x_test, (-1, 28 * 28))
    x_train, x_test = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_test, axis=2)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size=100)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=100)
    return db_train


def get_data_v3():
    # (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(numpy.float32) / 255., x_test.astype(numpy.float32) / 255.
    x_train, x_test = tf.reshape(x_train, (-1, 28 * 28)), tf.reshape(x_test, (-1, 28 * 28))
    x_train, x_test = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_test, axis=2)
    db_train = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=100)
    db_test = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=100)
    return db_train, db_test


class One_Conv(keras.Model):
    def __init__(self):
        super(One_Conv, self).__init__()
        self.sq1 = Sequential([
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[0], activation='relu', padding='same'),
            layers.BatchNormalization(),
            #layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            #layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[2], activation='relu', padding='same'),
            layers.BatchNormalization(),
            #layers.UpSampling1D(size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[3], activation='relu', padding='same'),
            layers.BatchNormalization(),
            #layers.UpSampling1D(size=2),
            layers.Conv1D(filters=input_data_dim, kernel_size=conv_kernel_size[4], activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        self.sq1.summary()
        return out1


def train(db_train, db_test, train_time=100):
    modle1 = One_Conv()
    modle1.build(input_shape=(None, 784, input_data_dim))
    loss_class = losses.CategoricalCrossentropy(from_logits=True)
    optimizer1 = tf.optimizers.Adam(lr=1e-3, name='SGD')
    for epoch1 in range(train_time):
        print('---->{0}'.format(epoch1))
        for step, x_train in enumerate(db_train):
            with tf.GradientTape() as tape:
                x_result = modle1(x_train)
                '''print(step, '========================x_result========================\n',
                      tf.reshape(x_result[20], [784]), '\n',
                      step, '========================x_train========================\n', tf.reshape(x_train[20], [784]))'''
                # loss1 = loss_class(x_train, x_result)
                loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_train, logits=x_result))
            grads = tape.gradient(loss1, modle1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, modle1.trainable_variables))
            if step % 100 == 0: print(epoch1, step, float(loss1))
        # 绘图部分
        x_test = next(iter(db_test))
        x_test_result = modle1(x_test)
        x_test_result = tf.reshape(tf.sigmoid(x_test_result), [-1, 28, 28])
        print(x_test_result)
        x_concat = tf.concat([tf.reshape(x_test, [-1, 28, 28]), x_test_result], axis=0)
        x_concat = x_concat.numpy() * 255
        x_concat = x_concat.astype(numpy.uint8)  # (200, 28, 28)
        save_images_compare(x_concat, 'cache/Conv1D_rec_epoch_%d.png' % epoch1)


def save_images_compare(imgs_numpy, path1):
    shape1 = imgs_numpy.shape
    image_all = Image.new('L', (2*shape1[1], 5*shape1[2]))
    index1 = 0
    for i1 in range(0, 2*shape1[1], shape1[1]):
        for i2 in range(0, 5*shape1[2], shape1[2]):
            image_single = Image.fromarray(imgs_numpy[index1], mode='L')
            image_all.paste(image_single, (i1, i2))
            index1 += 1
        index1 = int(shape1[0]/2)
    image_all.save(path1)
    pass


if __name__ == '__main__':
    db_train, db_test = get_data_v3()
    train(db_train=db_train, db_test=db_test)
