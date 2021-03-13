import tensorflow as tf
def learn1():
    a1 = tf.linspace(-6.,6.,10)
    print(tf.nn.tanh(a1))

learn1()