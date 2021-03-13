# SimpleRNN层功能验证
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# SimpleRNN测试
layer_1 = layers.SimpleRNN(64)
input1 = tf.random.normal([4, 80, 100])
out = layer_1(input1)
print(out.shape)

net1 = keras.Sequential([
    layers.SimpleRNN(64, return_sequences=True),
    layers.SimpleRNN(64)
])
out = net1(input1)
print(out.shape)