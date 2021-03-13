#测试Tensorflow的数字编码——onehot编码的转换
import tensorflow as tf

y = tf.constant([0,1,2,3])
y_hot = tf.one_hot(y , depth=10)
print('y_hot:',y_hot)