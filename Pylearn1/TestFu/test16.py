# SavedModel测试_失败
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets
from zh.model.utils import MNISTLoader


def learn1():
    num_epochs = 1
    batch_size = 50
    learning_rate = 0.001
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10), tf.keras.layers.Softmax()
    ])
    data_loader = MNISTLoader()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
    tf.saved_model.save(model, "saved/1")


def learn2():
    batch_size = 50
    model = tf.saved_model.load("saved/1")
    dataset_train, dataset_test, (x_train, y_train), (x_test, y_test) = load_data()
    model.evaluate(dataset_test)
    print("test accuracy: %f", )


if __name__ == '__main__':
    learn1()
