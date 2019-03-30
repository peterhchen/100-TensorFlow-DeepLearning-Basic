import tensorflow as tf

mynum = tf.constant(7)
mymat = tf.fill((3,3), 5)
myzer = tf.zeros((3,3))
myone = tf.ones((3,3))
rand1 = tf.random_normal((3,3),  mean=0.5,   stddev=1.0)
rand2 = tf.random_uniform((3,3), minval=0.5, maxval=1)

myops = [mynum, mymat, myzer, myone, rand1, rand2]

with tf.Session() as sess:
  for op in myops:
    print("op:",op, "value:",sess.run(op))
   #print("op:",op, "value:",op.eval())

#for Jupyter notebook (interactive session):
#sess = tf.InteractiveSession()
#for op in myops:
#  print("op:",op,"value:",sess.run(op))

