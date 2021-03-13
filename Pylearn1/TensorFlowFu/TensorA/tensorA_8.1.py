# Keras 高层接口
import tensorflow as tf
from tensorflow import keras  # 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets, metrics  # 导入常见网络层类


# 辅助用
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_train = tf.one_hot(y_train, depth=10)
    x_train = tf.reshape(x_train, (-1, 28 * 28))
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    y_test = tf.one_hot(y_test, depth=10)
    x_test = tf.reshape(x_test, (-1, 28 * 28))
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.batch(batch_size=200)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(batch_size=200)
    return dataset_train, dataset_test, (x_train, y_train), (x_test, y_test)


# softmax的调用
def learn1():
    x = tf.constant([2., 1., 0.1])
    layer = layers.Softmax(axis=-1)
    print(layer(x))


# 包装网络层成为一个整体
def learn2():
    network = Sequential([
        layers.Dense(3, activation=None),
        layers.ReLU(),
        layers.Dense(2, activation=None),
        layers.ReLU()
    ])
    x = tf.random.normal([4, 3])
    network(x)  # 从第一层开始，逐渐传播至最末层


# 追加网络层
def learn3():
    network = Sequential([])  # 先创建空的网络
    for i in range(3):
        network.add(layers.Dense(units=i * 10 + 1, activation='relu'))
    network.build(input_shape=(None, 6))
    print(network.summary())  # 显示容器中每层网络层的详细信息
    print('-------============++++++++++')
    print(network.variables)  # 显示容器中所有层的所有w b参数


# 模型装配
def learn4():
    # 0.参数准备
    dataset_train, dataset_test, (x_train, y_train), (x_test, y_test) = load_data()
    # 1.模型装配
    sq1 = Sequential([
        layers.Dense(256, activation='relu', input_shape=(None, 784)),  # input_shape=(None, 784)等同于input_dim=784
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    sq1.build()  # 创建网络并设置参数：输入数据规模为(None, 28*28)
    print(sq1.summary())
    sq1.compile(
        optimizer=optimizers.Adam(lr=0.01),  # 设置优化器为Adam，并设置优化器的学习率
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']  # 设置测量指标为准确率
    )

    # 2.模型训练
    # validation_data是验证测试数据集；validation_freq的意思是验证测试数据集的频率，即每过validation_freq个epoch时验证一次
    # train_db为训练集，val_db为验证集
    fit1 = sq1.fit(dataset_train, epochs=5, validation_data=dataset_test, validation_freq=2)
    print('HISTORY:', fit1.history)  # history函数会返回训练过程中的数据记录，包含loss、accuracy、val_loss、val_accuracy

    # 3.模型测试
    out_predict = sq1.predict(x_test[1:2, :])  # predict中的数据规模必须和sq1中build设置的规模一致：(None, 784)与[1 784]
    print(out_predict)  # 经过predict输出测试结果

    # 4.测试模型的性能
    print('SQ1:', sq1.evaluate(dataset_test))  # evaluate函数返回测试模型精度后的loss与accuracy

    # 5.保存当前网络参数——以张量方式保存
    path_save_weights = 'D:/boyLen/py/Pylearn1/dataLen/Keras/tensorA_8.1_save1_weights'
    sq1.save_weights(path_save_weights)  # 保存网络参数
    # del sq1  # 删除网络结构，测试保存的网络参数文件
    sq2 = Sequential([  # 重建网络结构
        layers.Dense(256, activation='relu'), layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(10)
    ])
    sq2.compile(
        optimizer=optimizers.Adam(lr=0.01), loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy']
    )
    sq2.load_weights(path_save_weights)     # 载入模型参数
    print('SQ2:', sq2.evaluate(dataset_test))

    # 6.保存当前网络参数——以网络方式保存
    path_save2 = 'D:/boyLen/py/Pylearn1/dataLen/Keras/tensorA_8.1_save2'
    sq1.save(path_save2)
    sq3 = tf.keras.models.load_model(path_save2)
    print('SQ3:', sq3.evaluate(dataset_test))

    # 7.保存当前网络参数——以无损SavedModel方式保存_【失败】
    '''
    path_save3 = 'D:/boyLen/py/Pylearn1/dataLen/Keras/tensorA_8.1_save3'
    tf.saved_model.save(sq1, path_save3)
    sq4 = tf.saved_model.load(path_save3)
    print('SQ4:', sq4.evaluate(dataset_test))'''

    # 8.继承Layer类
    class MyDense(layers.Layer):
        def __init__(self, input_dim, output_dim):  # 每一层的输入、输出维度。每次设定一层都要设置dim
            super(MyDense, self).__init__()
            # 继承了Layer后Layer的self变量都会继承过来
            self.kernel = self.add_variable('w', [input_dim, output_dim], trainable=True)  # add_weight、add_variable是Layer的API，是继承过来的
            self.bias = self.add_variable('b', [output_dim], trainable=True)  # self.kernel、self.bias是自己定义的名字

        def call(self, inputs, training=None):
            out = inputs @ self.kernel + self.bias
            return out

    # 9.继承Model类
    class MyModel(keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()     # 自定义层。自定义网络层，需要初始化__init__与call方法
            self.fc1 = MyDense(784, 256)
            self.fc2 = MyDense(256, 128)
            self.fc3 = MyDense(128, 64)
            self.fc4 = MyDense(64, 32)
            self.fc5 = MyDense(32, 10)

        def call(self, inputs, training=None):  # 前向运算逻辑
            x = tf.nn.relu(self.fc1(inputs))
            x = tf.nn.relu(self.fc2(x))
            x = tf.nn.relu(self.fc3(x))
            x = tf.nn.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    # 10.调用Model类
    my_model = MyModel()
    my_model.compile(
        optimizer=optimizers.Adam(lr=0.01), loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy']
    )
    fit2 = my_model.fit(dataset_train, epochs=5, validation_data=dataset_test, validation_freq=2)
    print('FIT2:', fit2)


# 加载现成通用模型
def learn5():
    # 1.加载ResNet50模型
    model1 = keras.applications.ResNet50(weights='imagenet', include_top=False)  # ResNet50的层数巨多
    model1.summary()
    # 2.设置池化层，将原数据[b,7,7,2048]池化为[b,2048]
    pool_layer = layers.GlobalAveragePooling2D()
    x1 = tf.random.normal([4, 7, 7, 2048])
    out_pool = pool_layer(x1)
    # 3.新建全连接层
    dense1 = layers.Dense(100)
    out_dense1 = dense1(out_pool)  # 全连接层与池化层相连接
    # 4.新建容器，将上面的三个层组合在一起
    sq1 = Sequential([model1, pool_layer, dense1])
    print(sq1.summary())


if __name__ == '__main__':
    learn4()
