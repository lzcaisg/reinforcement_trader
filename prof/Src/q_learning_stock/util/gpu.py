import tensorflow as tf
import time

def test():
    # Force execution on GPU #0 if available
    if tf.test.is_gpu_available():
        print("On GPU:0")
        with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
            x = tf.random_uniform([1000, 1000])
            assert x.device.endswith("GPU:0")
            _time_matmul(x)

    else:
        print("On CPU")
        with tf.device("CPU:0"):
            x = tf.random_uniform([1000, 1000])
            assert x.device.endswith("CPU:0")
            _time_matmul(x)

def _time_matmul(x):
    start = time.time()
    for _ in range(10):
        tf.matmul(x, x)

    result = time.time()-start

    print("10 loops: {:0.2f}ms".format(1000*result))