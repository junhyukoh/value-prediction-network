import scipy.signal
import numpy as np
import tensorflow as tf
from collections import namedtuple

def discount(x, gamma, time=None):
    if time is not None and time.size > 0:
        y = np.array(x, copy=True)
        for i in reversed(range(y.size-1)):
            y[i] += (gamma ** time[i]) * y[i+1]
        return y
    else:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

Batch = namedtuple("Batch", ["si", "a", "adv", "r", 
                            "terminal", "features", "reward", "step", "meta"])

def huber_loss(delta, sum=True):
    if sum:
        return tf.reduce_sum(tf.where(tf.abs(delta) < 1,
                        0.5 * tf.square(delta),
                        tf.abs(delta) - 0.5))
    else:
        return tf.where(tf.abs(delta) < 1,
                        0.5 * tf.square(delta),
                        tf.abs(delta) - 0.5)

def lower_triangular(x):
    return tf.matrix_band_part(x, -1, 0)

def to_bool(x):
    return x == 1

def parse_to_num(s):
    l = s.split(',')
    for i in range(0, len(l)):
        try:
            l[i] = int(l[i])
        except ValueError:
            l = []
            break
    return l
