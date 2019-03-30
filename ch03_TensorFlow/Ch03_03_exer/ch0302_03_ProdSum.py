import tensorflow as tf # tf-reduce-prod.py
x = tf.constant([100,200,300], name='x')
y = tf.constant([1,2,3], name='y')
sum_x  = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")
div_xy = tf.div(sum_x, prod_y, name="div_xy")

sess = tf.Session()
print(sess.run(sum_x))
print(sess.run(prod_y))
print(sess.run(div_xy))
sess.close()
# 600
# 6
# 100