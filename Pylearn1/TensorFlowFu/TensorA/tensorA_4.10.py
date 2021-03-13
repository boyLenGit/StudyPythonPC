# 前向传播实战-梯度下降算法-（反向传播隐含在Tensorflow梯度方法中）
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets
import tensorflow.keras as keras

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()  # 自动下载MNIST数据并返回两个numpy.array对象，前两个是训练集，后两个是测试集.x都是图片数据集，y都是对应图片的编号。下载的数据为numpy格式的文件
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 将图片x数据转化为张量，并缩放到(1,-1)之间
    y = tf.convert_to_tensor(y, dtype=tf.int32)  # 将图片编号y转化为张量
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    x = tf.reshape(x, (-1, 28 * 28))  # 改变视图， [b, 28, 28] => [b, 28*28]，即打平
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset.数据集对象
    train_dataset = train_dataset.batch(batch_size=200)  # 将数据集将分为300个Batch，每个Batch有200个样本，让Tensorflow一次性处理200份,以加快训练速度，一次跑的数据量不会过大，防止爆内存
    # 原本y是y.shape: (60000, 10)，经过batch(200)后变为y.shape: (200, 10)；x则由(60000, 28, 28)先经reshape变为(60000, 784)再经batch变为(200, 784)
    return train_dataset  # train_dataset中包含200一组的数据N组（N=60000/200),读取的话需要用step, (x, y) in enumerate(train_dataset1)读取出来


def init_paramaters():
    # 每层的张量都需要被优化，故使用 Variable 类型，并使用截断的正太分布初始化权值张量     偏置向量初始化为 0 即可
    # 第一层的参数
    w1 = tf.Variable(
        tf.random.truncated_normal([784, 256], stddev=0.1))  # [784,256]的原因是图片数据是[b, 784]，要符合矩阵的乘法规则[b, 784]@[784, 256]
    b1 = tf.Variable(tf.zeros([256]))
    # 第二层的参数
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 第三层的参数
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):  # 只训练一遍，共60000组数据
    listInclude_w_b = []
    for step, (x, y) in enumerate(train_dataset):  # enumerate将train_dataset的数据读出来，自动生成( step (200条x) (200条y) )的list的索引序列，每一个step有200条，也即一次循环，直到将所有数据读满（60000条）
        with tf.GradientTape() as tape:  # 用于保存计算模型的信息，便于反复求导计算
            h1 = x @ w1 + b1  # 第一层计算， [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
            h1 = tf.nn.relu(h1)  # 通过激活函数relu，构成非线性关系。tf.nn是一个tf库，提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
            h2 = h1 @ w2 + b2  # 第二层计算， [b, 256] => [b, 128]
            h2 = tf.nn.relu(h2)  # 通过激活函数relu，构成非线性关系
            out = h2 @ w3 + b3  # 输出层计算， [b, 128] => [b, 10]
            # out = tf.nn.softmax(out, axis=1)
            # loss_square = tf.square(y - out)   #计算网络输出与标签之差 v2 的平方
            # loss_square = keras.losses.mse(y,out)
            # loss = tf.reduce_mean(loss_square)# 求阵列loss的总均值，由矩阵得到一个数值，为均方差（标准差）
            loss = keras.losses.MeanSquaredError()
            loss = loss(y, out)
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3,
                                         b3])  # 【反向传播】自动更新梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]。gradient作用：返回列表表示各个变量的梯度值，和source中的变量列表一一对应，表明这个变量的梯度。
        # 【梯度下降法】梯度更新， assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub(lr * grads[0])  # assign_sub是对w1进行更新，减去lr * grads[0]
        b1.assign_sub(lr * grads[1])  # 不能使用w1-=lr * grads[0]来替代这个,原因未知（变为tf张量？）
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step % 200 == 0 and step != 0:
            print('epoch=', epoch, ';step=', step, ';loss:', loss.numpy())
        listInclude_w_b = [w1, b1, w2, b2, w3, b3]
    return loss.numpy(), listInclude_w_b


def TestModul(listInclude_w_b, epochs):  # 测试模型精度的模块
    w1 = listInclude_w_b[0]
    b1 = listInclude_w_b[1]
    w2 = listInclude_w_b[2]
    b2 = listInclude_w_b[3]
    w3 = listInclude_w_b[4]
    b3 = listInclude_w_b[5]
    EqualResult = []
    TestOfOneHot = []
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x_val, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    x = tf.reshape(x, (-1, 28 * 28))
    datasetForTestModul = tf.data.Dataset.from_tensor_slices((x, y))
    datasetForTestModul = datasetForTestModul.batch(batch_size=200)
    cnt_equal = 0
    cnt_FullOfCal = 0
    for step, (x, y) in enumerate(datasetForTestModul):
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3
        # 计算out与y的匹配率，得到模型的准确率
        ProbabilityOfOut = tf.nn.softmax(out, axis=1)
        TestIDOfOneHot = tf.argmax(ProbabilityOfOut, axis=1)
        EqualResult = tf.equal(TestIDOfOneHot, tf.argmax(y, axis=1))
        cnt_FullOfCal += 1
        for i1 in EqualResult:
            if tf.equal(tf.constant(True), i1): cnt_equal += 1
    print('神经网络模型准确率：', 100 * cnt_equal / (200 * cnt_FullOfCal), '%', '。训练次数：', epochs, '，总数据数：', cnt_FullOfCal * 200,
          ',其中准确预测数：', cnt_equal)


def train(epochs):
    losses = []
    listInclude_w_b = []
    train_dataset = load_data()  # 虽然train_dataset的shapes为((None, 784), (None, 10))，实质上还是存有200一组共60000条数据
    w1, b1, w2, b2, w3, b3 = init_paramaters()
    for epoch in range(epochs):
        loss, listInclude_w_b = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001)
        losses.append(loss)
    # 测试模型准确度
    TestModul(listInclude_w_b, epochs)
    # 绘制曲线
    x = [i for i in range(0, epochs)]
    plt.plot(x, losses, color='blue', marker='s', label='训练')
    plt.title('前向传播算法')
    plt.xlabel('Epoch训练次数')
    plt.ylabel('loss均方差')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线3.png')
    plt.close()


if __name__ == '__main__':
    train(epochs=30)  # 设定训练次数
