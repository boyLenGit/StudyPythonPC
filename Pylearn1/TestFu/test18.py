# TensorBoard的使用，共包括两种方法：第一种是高层版，与Sequential结合用；第二种底层版，基于tf.GradientTape()的tf.summary方法
import tensorflow as tf
import datetime


# 方法1：与Sequential相结合
def learn1():
    # 1.获取数据
    data_mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # 2.定义网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 3.定义TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # tensorboard_callback回调可确保创建和存储日志
    model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
    # CMD命令：tensorboard --logdir=D:\boyLen\py\Pylearn1\logs\fit\
    # 浏览器：http://localhost:6006/


# =====================================================================================================================
# 方法2：tf.GradientTape()的tf.summary
def learn2():
    # 1.获取数据
    data_mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(60000).batch(64)
    test_dataset = test_dataset.batch(64)

    # 2.定义loss、mean、accuracy对象，用的时候只需调用对象即可
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    # 3.定义路径名，创建summary（train与test分别各自创建summary）
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # 4.定义模型、训练函数、测试函数
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    def train_step(model, optimizer, x_train, y_train):  # 训练的FP、BP
        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = loss_object(y_train, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # ！optimizer求梯度的写法
        train_loss(loss)  # 调用tf.keras.metrics.Mean对象
        train_accuracy(y_train, predictions)

    def test_step(model, x_test, y_test):  # 测试的FP、BP
        predictions = model(x_test)
        loss = loss_object(y_test, predictions)
        test_loss(loss)
        test_accuracy(y_test, predictions)

    # 5.开始训练
    EPOCHS = 5  # 训练次数
    for epoch in range(EPOCHS):
        for (x_train, y_train) in train_dataset:
            train_step(model, optimizer, x_train, y_train)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_dataset:
            test_step(model, x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))

        # 每个循环后要清空metrics
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    # tensorboard --logdir=D:\boyLen\py\Pylearn1\logs\gradient_tape\
    # 浏览器：http://localhost:6006/


if __name__ == '__main__':
    learn2()
