# sec(60 degree)
import tensorflow as tf #tf-exer01.py
import math as m
PI = tf.constant(m.pi)

sess = tf.Session()
print('sec(2/3*PI): ', sess.run(1/tf.cos(m.pi*2.0/3.0)))

# cot(60 degree)
print('cot(2/3*PI): ', sess.run(1/tf.tan(m.pi*2.0/3.0)))