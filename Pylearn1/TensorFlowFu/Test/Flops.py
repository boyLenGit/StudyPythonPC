
import tensorflow as tf

g = tf.Graph()
run_meta = tf.compat.v1.RunMetadata()
with g.as_default():
    A = tf.random.uniform((5, 5))
    B = tf.random.uniform((5, 5))
    C = tf.matmul(A, B)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    if flops is not None:
        print('TF stats gives', flops.total_float_ops)