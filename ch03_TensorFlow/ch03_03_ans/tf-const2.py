import tensorflow as tf # tf-const2.py

aconst = tf.constant(3.0)
print(aconst)

# Automatically close "sess"
with tf.Session() as sess:
  print(sess.run(aconst)) 

