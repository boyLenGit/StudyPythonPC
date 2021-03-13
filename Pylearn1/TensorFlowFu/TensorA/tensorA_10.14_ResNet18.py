# ResNet18实战，对比CIFAR10
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
import datetime

# TensorBoard的路径参数声明
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape_test/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# tensorboard --logdir=D:\boyLen\py\Pylearn1\logs\gradient_tape_test\ --host=192.168.3.22
# %tensorboard --logdir=D:\boyLen\py\Pylearn1\logs\gradient_tape2\ --host=180.201.161.93
# 浏览器：http://192.168.3.22:6006/

gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取所有 GPU 设备列表
if gpus:
    try:
        for gpu in gpus:  # 设置 GPU 显存占用为按需分配
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


def get_data(batch_num=110):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()  # 在线下载，加载 CIFAR10 数据集
    y_train = tf.squeeze(y_train, axis=1)  # 删除 y 的一个维度，[b,1] => [b]
    y_test = tf.squeeze(y_test, axis=1)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # 构建训练集对象
    train_db = train_db.shuffle(1000).map(preprocess).batch(batch_num)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # 构建测试集对象
    test_db = test_db.map(preprocess).batch(batch_num)
    x_train = tf.cast(x_train, dtype=tf.float32)
    return x_train, y_train, x_test, y_test, train_db, test_db  # x维度：(50000, 32, 32, 3)


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
        self.sq4.add(layers.Dense(512, activation=tf.nn.relu))
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
    loss_class = losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(learning_rate=0.001, name='SGD')
    # 训练模型，并且每个epoch测试一次模型精度
    for epoch1 in range(1):
        for step1, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape() as tape1:
                out = resnet_sq(x_train, training=True)  # training=True是让BN层区分是训练模型还是测试模型，从而实现仅在训练时BN化
                y_train_onehot = tf.one_hot(y_train, depth=10)
                loss = loss_class(y_train_onehot, out)
                list_loss.append(loss)
            grads = tape1.gradient(loss, resnet_sq.trainable_variables)
            optimizer.apply_gradients(zip(grads, resnet_sq.trainable_variables))
            if step1 % 20 == 0:
                print(step1, end='|')
            with train_summary_writer.as_default():  # 调用TensorBoard
                tf.summary.scalar('LossLite_ResNet18', loss, step=step1)
        # 测试准确度
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
        with train_summary_writer.as_default():  # 调用TensorBoard
            tf.summary.scalar('Loss_ResNet18', loss, step=epoch1)
            tf.summary.scalar('Accuracy_ResNet18', predict_result, step=epoch1)
    return list_loss


def plot_tensorboard(list_loss):
    image_name = 'D:/boyLen/Python/Pylearn1/image/image_tb_resnet_0'
    plt.plot(range(len(list_loss)), list_loss, color='blue')
    plt.title("ResNet18-boyLen")
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.savefig(image_name + '.png')
    plt.close()
    image_file = tf.io.read_file(image_name + '.png')
    image_tensor = tf.image.decode_png(image_file)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    with train_summary_writer.as_default():  # 调用TensorBoard
        tf.summary.image('Loss', image_tensor, step=0)


if __name__ == '__main__':
    x__train, y__train, x__test, y__test, train__db, test__db = get_data()
    print('GetData数据规模：', tf.shape(x__train).numpy(), tf.shape(y__train).numpy())
    list_loss_plot = train_resnet(x__train, y__train, x__test, y__test, train__db, test__db)
    plot_tensorboard(list_loss_plot)

