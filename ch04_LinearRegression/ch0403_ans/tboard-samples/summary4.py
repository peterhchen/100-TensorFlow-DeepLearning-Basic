#https://itnext.io/how-to-use-tensorboard-5d82f8654496
# => tensorboard and images 
import tensorflow as tf

# clear the defined variables and operations of previous cell
tf.reset_default_graph()   

# create the variables
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 0: reshape it to 4D-tensors
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1))
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3))

# step 1: create the summaries
gs_summary = tf.summary.image('Grayscale', w_gs_reshaped)
c_summary = tf.summary.image('Color', w_c_reshaped, max_outputs=5)

# step 2: merge all summaries
merged = tf.summary.merge_all()

# create the op for initializing all variables
init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  # step 3: creating the writer inside the session
  writer = tf.summary.FileWriter('./summary4', sess.graph)

  # initialize all variables
  sess.run(init)

  # step 4: evaluate the merged op to get the summaries
  summary = sess.run(merged)

  # step 5: add summary to the writer 
  writer.add_summary(summary)
  print('Done writing the summaries')

