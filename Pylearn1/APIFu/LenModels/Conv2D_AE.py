import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential


class simple_cnn_ae(keras.Model):
    def __init__(self):
        """
        结构：Conv2D-Conv2D-Conv2D-Dense-Dense-Dense-Conv2D-Conv2D-Conv2D
        """
        super(simple_cnn_ae, self).__init__()
        self.sq1 = Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'),
            layers.Dense(32 * 32, activation='relu'),
            layers.Dense(16 * 16, activation='relu'),
            layers.Dense(32 * 32, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1, kernel_size=[8, 8], activation='tanh', padding='same'),
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


class Helper_GoogleNet_Inception_FloorA(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = Sequential([
            layers.MaxPool2D(),
            layers.Conv2D(filters=filters, kernel_size=[8, 8], activation='relu', padding='same')
        ])
        self.branch2 = Sequential([
            layers.MaxPool2D(),
            layers.Conv2D(filters=filters, kernel_size=[4, 4], activation='relu', padding='same')
        ])
        self.branch3 = Sequential([
            layers.Conv2D(filters=filters, kernel_size=[2, 2], activation='relu', padding='same'),
            layers.MaxPool2D()
        ])

    def call(self, inputs, **kwargs):
        out_branch1 = self.branch1(inputs)
        out_branch2 = self.branch2(inputs)
        out_branch3 = self.branch3(inputs)
        out_fusion = layers.concatenate([out_branch1, out_branch2, out_branch3])
        return out_fusion


class Helper_GoogleNet_Inception_FloorB(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = Sequential([
            layers.MaxPool2D(),
            layers.Conv2D(filters=filters, kernel_size=[2, 2], activation='relu', padding='same'),
            layers.UpSampling2D()
        ])
        self.branch2 = layers.Conv2D(filters=filters, kernel_size=[2, 2], activation='relu', padding='same')

    def call(self, inputs, **kwargs):
        out_branch1 = self.branch1(inputs)
        out_branch2 = self.branch2(inputs)
        out_fusion = layers.concatenate([out_branch1, out_branch2])
        return out_fusion


class Conv2DAE_GoogleNet_v1(keras.Model):
    def __init__(self):
        """
        结构：
        """
        super(Conv2DAE_GoogleNet_v1, self).__init__()
        self.sq1 = Sequential([
            layers.BatchNormalization(),
            Helper_GoogleNet_Inception_FloorA(filters=32),
            layers.BatchNormalization(),
            Helper_GoogleNet_Inception_FloorA(filters=64),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'),
            layers.Dense(32 * 32, activation='relu'),
            layers.Dense(16 * 16, activation='relu'),
            layers.Dense(32 * 32, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1, kernel_size=[8, 8], activation='tanh', padding='same'),
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


class Conv2DAE_GoogleNet_v2(keras.Model):
    def __init__(self):
        """
        结构：
        """
        super(Conv2DAE_GoogleNet_v2, self).__init__()
        self.sq1 = Sequential([
            layers.BatchNormalization(),
            Helper_GoogleNet_Inception_FloorA(filters=32),
            layers.BatchNormalization(),
            Helper_GoogleNet_Inception_FloorA(filters=32),
            layers.BatchNormalization(),
            Helper_GoogleNet_Inception_FloorB(filters=32),
            layers.Dense(32 * 32, activation='relu'),
            layers.Dense(16 * 16, activation='relu'),
            layers.Dense(32 * 32, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=[4, 4], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=[8, 8], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1, kernel_size=[8, 8], activation='tanh', padding='same'),
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1
