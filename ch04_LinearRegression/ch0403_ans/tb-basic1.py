import tensorflow as tf
import numpy as np

step = 0
session = tf.Session()

tensorboardVar = tf.Variable(0, "tensorboardVar")

pythonVar = tf.placeholder("int32", [])

update_tensorboardVar = tensorboardVar.assign(pythonVar)
tf.summary.scalar("myVar", update_tensorboardVar)

merged = tf.summary.merge_all()

sum_writer = tf.summary.FileWriter('/tmp/train/c/', session.graph)

session.run(tf.global_variables_initializer())

for i in range(100):
  #_, result = session.run([update_tensorboardVar, merged])
  j = np.array(i)
  _, result = session.run([update_tensorboardVar, merged], feed_dict={pythonVar: j})
  sum_writer.add_summary(result, step)
  step += 1

