import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

point_count = 100000
x_data = np.linspace(0.0, 10.0, point_count)
print("x_data shape:",x_data.shape)
print("x_data len:",len(x_data))
noise  = np.random.randn(len(x_data))
print("noise shape:",noise.shape)

#y = m*x + b: m = 0.5 and b = 5 
y_true = (0.5 * x_data) + 5 + noise 

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

print("top 5 x_df:")
print(x_df.head())
print("top 5 y_df:")
print(y_df.head())

my_data = pd.concat([x_df, y_df], axis=1)
print("top 5 my_data:")
print(my_data.head())

batch_size = 8

# initialize m and b with random values 
arr1 = np.random.rand(2) # arr1 = (m,b)
m = tf.Variable(arr1[0])
b = tf.Variable(arr1[1])

xph = tf.placeholder(tf.float64, [batch_size])
yph = tf.placeholder(tf.float64, [batch_size])

# define the linear model:
y_model = m * xph + b

# define the error function:
error = tf.reduce_sum(tf.square(yph-y_model))

# define the optimizer function:
lr = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op  = optimizer.minimize(error)

with tf.Session() as sess:
  tf.global_variables_initializer().run()

  batches = 1000
  for i in range(batches):
    rand_ind = np.random.randint(len(x_data), size=batch_size)
    feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]} 
    sess.run(train_op, feed_dict = feed)

  final_m, final_b = sess.run([m,b])

print("fm:",final_m,"fb:",final_b)

# select 250 points to plot:
subd = my_data.sample(n=250)
subd.plot(kind='scatter', x='X Data', y='Y',color='blue')

# plot best-fitting line:
x_test = np.linspace(-1,11,10)
y_hat  = final_m*x_test + final_b
plt.plot(x_test, y_hat, 'r')

plt.show()

