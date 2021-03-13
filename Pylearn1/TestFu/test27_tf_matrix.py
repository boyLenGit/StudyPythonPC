import tensorflow as tf

a1_1 = tf.constant([
    [[0, 0, 0],
     [0, 100, 0],
     [0, 0, 0]]
])
a1_2 = tf.constant([
    [[0, 0, 0],
     [0, 100, 100],
     [0, 0, 0]]
])
a1_3 = tf.constant([
    [[0, 0, 100],
     [0, 100, 0],
     [0, 0, 0]]
])
a2_1 = tf.constant([
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
])
a2_2 = [
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
]

a3_1 = tf.constant([
    [1]
])
a3_2 = tf.constant([
    [1, 1, 1]
])

a4_1 = [[1, 2, 3, 0, 0],
        [4, 0, 5, 0, 0],
        [6, 7, 8, 0, 0]]

print('@ ', a4_1[:][:])
print('@ ', a4_1[0])

print(a1_1 * a2_2)
print(a1_2 * a2_2)
print(tf.reduce_max(a1_3 * a2_1) != 0)
if tf.reduce_max(a1_3 * a2_1) != 0: print('YEs')
print(tf.shape(a1_2))
