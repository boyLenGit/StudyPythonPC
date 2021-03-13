# 实现论文内容
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import tensorflow.keras.datasets as datasets

def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255., tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_train = tf.reshape(x_train, (-1, 28 * 28))  # 数据规模(?, 784)
    print('-----shape1:', tf.shape(x_train), x_train.shape)
    y_train = tf.one_hot(y_train, depth=10)
    x_train = tf.stack([x_train, x_train], axis=1)
    print('-----shape2:', tf.shape(x_train), x_train.shape)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size=100)
    return db_train


class One_Conv(keras.Model):
    def __init__(self):
        super(One_Conv, self).__init__()
        self.sq1 = Sequential([
            layers.Conv1D(filters=40, kernel_size=130, activation='relu', padding='same'),
            layers.MaxPool1D(pool_size=2, padding='same'),
            layers.Conv1D(filters=40, kernel_size=65, activation='relu', padding='same'),
            layers.MaxPool1D(pool_size=2, padding='same'),
            layers.Conv1D(filters=40, kernel_size=65, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=40, kernel_size=65, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=2, kernel_size=130, activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        print('-----shape4:', tf.shape(inputs), inputs.shape)
        out1 = self.sq1(inputs)
        return out1


def train(db_train, train_time=100):
    model1 = One_Conv()
    model1.build(input_shape=(1, 2, 784))
    print ('-----shape5:')
    model1.summary()
    optimizer1 = tf.optimizers.Adam(lr=1e-3)
    for epoch1 in range(train_time):
        for step, (x_train, y_train) in enumerate(db_train):
            with tf.GradientTape() as tape:
                print('-----shape3:', tf.shape(x_train), x_train.shape)
                x_result = model1(x_train)
                print('-----shape6:', tf.shape(x_train), x_train.shape, x_result.shape)
                loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_train, logits=x_result)
                loss1 = tf.reduce_mean(loss1)
            grads = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 100 == 0: print(epoch1, step, float(loss1))



if __name__ == '__main__':
    db_train = get_data()
    train(db_train=db_train)