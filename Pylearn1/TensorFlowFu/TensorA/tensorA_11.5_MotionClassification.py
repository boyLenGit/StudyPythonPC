# SimpleRNNCell层
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import matplotlib.pyplot as plt

batch_num, total_vocabulary, max_sentence_len, embedding_len = 128, 10000, 80, 100


def f_get_data():
    # 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
    # num_words=10000表示保留前1w个常出现的单词，num_words表示词汇表大小
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_vocabulary)


    # 查看数字编码表
    word_index1 = keras.datasets.imdb.get_word_index()
    '''
    for i0, (i1, i2) in enumerate(word_index1.items()):
        print('word_index_{0}:'.format(i0), i1, i2)
        if i0 > 20: break
    '''

    # 翻转数字编码表
    word_index2 = {i1: (i2 + 3) for i1, i2 in word_index1.items()}
    word_index2['<PAD>'] = 0  # 标识符
    word_index2['<START>'] = 1  # 起始标识符
    word_index2['<UNK>'] = 2  # 标识符
    word_index2['<UNUSED>'] = 3  # 标识符
    word_index3_reverse = dict([(value, key) for (key, value) in word_index2.items()])
    print('键-值翻转后的word_index3:', (list(word_index3_reverse.items()))[:10])  # .item是字典的API

    def f_decode_review(input_text):  # 将数字编码转换为字符串格式的单词
        return ' '.join([word_index3_reverse.get(i, '?') for i in input_text])

    print('-------------------------------在这里观察训练集本质-------------------------------')
    print('转换前的train数据1：', x_train[0])  # 可见训练集是由许多句子构成的：[句子1，句子2，句子3...]，每个句子中又有多个单词
    print('转换后的train数据1：', f_decode_review(x_train[0]))
    print('转换前的train数据2：', x_train[1])
    print('转换后的train数据2：', f_decode_review(x_train[1]))
    print('转换前的train数据3：', y_train, '0代表消极，1代表积极')

    # -------------------------------准备训练数据-------------------------------
    # 句子最大长度s=100，大于s的句子将会被截断，小于的将填充
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_sentence_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_sentence_len)

    # 构建成RNN的数据集
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size=128, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(batch_size=128, drop_remainder=True)
    print('构建数据集，shape:', x_train.shape, x_test.shape)  # 结果：(25000, 100) (25000, 100)，其中25000是句子个数，100是句子长度
    return db_train, db_test


class MyRNN(keras.Model):
    def __init__(self, units1):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batch_num, units1])]
        self.state1 = [tf.zeros([batch_num, units1])]
        # 这里的“input_dim=total_vocabulary”要与load_data中的“num_words=total_vocabulary”规模相一致
        # 这里的“input_length=max_sentence_len”
        self.embedding = layers.Embedding(input_dim=total_vocabulary, output_dim=embedding_len, input_length=max_sentence_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units=units1, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units=units1, dropout=0.5)
        # 分类网络完成二分类任务，故输出节点设置为1
        self.out_layer = Sequential([layers.Dense(units1), layers.Dropout(rate=0.5), layers.ReLU(), layers.Dense(1)])

    def call(self, inputs, training=None):
        x1 = self.embedding(inputs)  # [b,80]-->[b,80,100]
        # [b,80,100]-->[b,64]
        state0, state1 = self.state0, self.state1
        # SimpleRNN层
        for i1 in tf.unstack(x1, axis=1):  #
            out0, state0 = self.rnn_cell0(i1, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        # 全连接层
        x1 = self.out_layer(out1, training)
        return tf.sigmoid(x1)

def train_test(epochs=20):
    db_train, db_test = f_get_data()
    model1 = MyRNN(units1=64)  # units1=64状态向量长度
    model1.compile(optimizer=optimizers.Adam(0.001), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
    model1.fit(db_train, epochs=epochs, validation_data=db_test)
    result_test = model1.evaluate(db_test)
    print(result_test)


if __name__ == '__main__':
    train_test()
