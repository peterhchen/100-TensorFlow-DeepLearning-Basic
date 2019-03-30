#https://itnext.io/how-to-use-tensorboard-5d82f8654496
# => tensorboard and merged data 
import tensorflow as tf

# clear the defined variables and operations of previous cell
tf.reset_default_graph()   

# create the variables
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1: create the summaries
# A scalar summary for the scalar tensor
scalar_summary = tf.summary.scalar('My_scalar_summary', x_scalar)

# A histogram summary for the non-scalar (i.e. 2D or matrix) tensor
histogram_summary = tf.summary.histogram('My_histogram_summary', x_matrix)

# step 2: merge all summaries
merged = tf.summary.merge_all()

# create the op for initializing all variables
init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  # step 3: creating the writer inside the session
  writer = tf.summary.FileWriter('./summary3', sess.graph)

  for step in range(100):
    # loop over several initializations of the variable
    sess.run(init)

    # step 4: evaluate the merged summaries
    summary = sess.run(merged)

    # step 5: add summary to the writer 
    writer.add_summary(summary, step)
  print('Done writing the summaries')

