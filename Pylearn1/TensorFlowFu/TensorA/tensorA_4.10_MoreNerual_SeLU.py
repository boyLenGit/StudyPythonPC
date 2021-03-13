#前向传播实战-梯度下降算法-7层神经网络（反向传播隐含在Tensorflow梯度方法中）
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets

def load_data():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()  #自动下载MNIST数据并返回两个numpy.array对象，前两个是训练集，后两个是测试集.x都是图片数据集，y都是对应图片的编号。下载的数据为numpy格式的文件
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  #将图片x数据转化为张量，并缩放到(1,-1)之间
    y = tf.convert_to_tensor(y, dtype=tf.int32)  #将图片编号y转化为张量
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    x = tf.reshape(x, (-1, 28 * 28))  # 改变视图， [b, 28, 28] => [b, 28*28]，即打平
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建Dataset.数据集对象
    train_dataset = train_dataset.batch(batch_size=200)  #将数据集将分为300个Batch，每个Batch有200个样本，让Tensorflow一次性处理200份,以加快训练速度，一次跑的数据量不会过大，防止爆内存
                                              #原本y是y.shape: (60000, 10)，经过batch(200)后变为y.shape: (200, 10)；x则由(60000, 28, 28)先经reshape变为(60000, 784)再经batch变为(200, 784)
    return train_dataset   #train_dataset中包含200一组的数据N组（N=60000/200),读取的话需要用step, (x, y) in enumerate(train_dataset1)读取出来

def init_paramaters():
    w1 = tf.Variable(tf.random.truncated_normal([784, 500], stddev=0.1))
    b1 = tf.Variable(tf.zeros([500]))
    w2 = tf.Variable(tf.random.truncated_normal([500, 384], stddev=0.1))
    b2 = tf.Variable(tf.zeros([384]))
    w3 = tf.Variable(tf.random.truncated_normal([384, 256], stddev=0.1))
    b3 = tf.Variable(tf.zeros([256]))
    w4 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b4 = tf.Variable(tf.zeros([128]))
    w5 = tf.Variable(tf.random.truncated_normal([128, 50], stddev=0.1))
    b5 = tf.Variable(tf.zeros([50]))
    w6 = tf.Variable(tf.random.truncated_normal([50, 10], stddev=0.1))
    b6 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6

def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, lr=0.001):#只训练一遍，共60000组数据
    listInclude_w_b=[]
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            h1 = x  @ w1 + tf.broadcast_to(b1, (x.shape[0], 500))
            h1 = tf.nn.selu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.selu(h2)
            h3 = h2 @ w3 + b3
            h3 = tf.nn.selu(h3)
            h4 = h3 @ w4 + b4
            h4 = tf.nn.selu(h4)
            h5 = h4 @ w5 + b5
            h5 = tf.nn.selu(h5)
            out = h5 @ w6 + b6
            out = tf.nn.softmax(out, axis=1)
            loss_square = tf.square(y - out)
            loss = tf.reduce_mean(loss_square)
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6])
        w1.assign_sub(lr * grads[0]);        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2]);        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4]);        b3.assign_sub(lr * grads[5])
        w4.assign_sub(lr * grads[6]);        b4.assign_sub(lr * grads[7])
        w5.assign_sub(lr * grads[8]);        b5.assign_sub(lr * grads[9])
        w6.assign_sub(lr * grads[10]);        b6.assign_sub(lr * grads[11])
        if step % 200 == 0 and step != 0:
            print('epoch=',epoch,';step=', step, ';loss:', loss.numpy())
        listInclude_w_b = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6]
    return loss.numpy(),listInclude_w_b

def TestModul(listInclude_w_b,epochs):
    w1 = listInclude_w_b[0];b1 = listInclude_w_b[1];w2 = listInclude_w_b[2];b2 = listInclude_w_b[3];w3 = listInclude_w_b[4];b3 = listInclude_w_b[5]
    w4 = listInclude_w_b[6];    b4 = listInclude_w_b[7];    w5 = listInclude_w_b[8];    b5 = listInclude_w_b[9];    w6 = listInclude_w_b[10];    b6 = listInclude_w_b[11]
    EqualResult = [];TestOfOneHot=[]
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x_val, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    x = tf.reshape(x, (-1, 28 * 28))
    datasetForTestModul = tf.data.Dataset.from_tensor_slices((x, y))
    datasetForTestModul = datasetForTestModul.batch(batch_size=200)
    cnt_equal = 0;cnt_FullOfCal = 0
    for step, (x, y) in enumerate(datasetForTestModul):
        with tf.GradientTape() as tape:
            h1 = x @ w1 + tf.broadcast_to(b1, (x.shape[0], 500))
            h1 = tf.nn.selu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.selu(h2)
            h3 = h2 @ w3 + b3
            h3 = tf.nn.selu(h3)
            h4 = h3 @ w4 + b4
            h4 = tf.nn.selu(h4)
            h5 = h4 @ w5 + b5
            h5 = tf.nn.selu(h5)
            out = h5 @ w6 + b6
        #计算out与y的匹配率，得到模型的准确率
        ProbabilityOfOut = tf.nn.softmax(out, axis=1)
        TestIDOfOneHot = tf.argmax(ProbabilityOfOut, axis=1)
        EqualResult = tf.equal(TestIDOfOneHot, tf.argmax(y, axis=1))
        cnt_FullOfCal += 1
        for i1 in EqualResult:
            if tf.equal(tf.constant(True), i1): cnt_equal += 1
    print('神经网络模型准确率：',100*cnt_equal/(200*cnt_FullOfCal),'%','。训练次数：',epochs,'，总数据数：',cnt_FullOfCal*200,',其中准确预测数：',cnt_equal)

def train(epochs):
    losses = [];listInclude_w_b=[]
    train_dataset = load_data()  #虽然train_dataset的shapes为((None, 784), (None, 10))，实质上还是存有200一组共60000条数据
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = init_paramaters()
    for epoch in range(epochs):
        loss,listInclude_w_b = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, lr=0.001)
        losses.append(loss)
    #测试模型准确度
    TestModul(listInclude_w_b,epochs)
    # 绘制曲线
    x = [i for i in range(0, epochs)]
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = ['STKaiti']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(x, losses, color='blue', marker='s', label='训练')
    plt.title('前向传播算法')
    plt.xlabel('Epoch训练次数')
    plt.ylabel('loss均方差')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线3.png')
    plt.close()

if __name__ == '__main__':
    train(epochs=200)  #设定训练次数