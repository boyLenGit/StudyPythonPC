# LeNet5实战
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


def get_data_minst():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # 将图片x数据转化为张量，并缩放到(1,-1)之间
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)  # 将图片编号y转化为张量
    y_train = tf.one_hot(y_train, depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size=200)
    return x_train, y_train, x_train, x_test, y_train, y_test, train_dataset


def main(train_dataset):
    # 1.新建网络层容器
    sq1 = Sequential([  # 网络容器
        layers.Conv2D(filters=6, kernel_size=3, strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        layers.ReLU(),  # 激活函数
        layers.Conv2D(filters=16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        layers.ReLU(),  # 激活函数
        layers.Flatten(),  # 打平层，方便全连接层处理
        layers.Dense(120, activation='relu'),  # 全连接层，120 个节点
        layers.Dense(84, activation='relu'),  # 全连接层，84 节点
        layers.Dense(10)  # 全连接层，10 个节点
    ])
    sq1.build(input_shape=(4, 28, 28, 1))
    sq1.summary()
    # 2.训练模型
    loss_class = losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(learning_rate=0.001, name='SGD')
    list_loss = []
    for epoch in range(50):  # 训练次数
        for step1, (x_train, y_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:  # 构建梯度记录环境
                x_train = tf.expand_dims(x_train, axis=3)  # 插入通道维度，=>[b,28,28,1]
                # training=True是给BP层用的
                out = sq1(x_train, training=True)  # 前向计算，获得 10 类别的概率分布，[b, 784] => [b, 10]
                loss = loss_class(y_train, out)  # 计算交叉熵损失函数，标量
            grads = tape.gradient(loss, sq1.trainable_variables)  # 自动计算梯度
            optimizer.apply_gradients(zip(grads, sq1.trainable_variables))  # 自动更新参数
            if step1 % 10 == 0: print(step1, end='|')
        list_loss.append(loss.numpy())
        print('EPOCH:', epoch, 'LOSS:', loss.numpy())
    plt.plot(range(len(list_loss)), list_loss)
    plt.show()


def control_center():
    x, y, x_train, x_test, y_train, y_test, train_dataset = get_data_minst()
    main(train_dataset)


if __name__ == '__main__':
    control_center()
