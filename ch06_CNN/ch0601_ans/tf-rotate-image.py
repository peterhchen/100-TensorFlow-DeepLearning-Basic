import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plot

myfile = "sample1.png"
myimage = img.imread(myfile)
image = tf.Variable(myimage,name='image')
vars = tf.global_variables_initializer()

sess = tf.Session()
# orignal=[0, 1, 2]
# 0: horizontal, 1: vertical
#flipped = tf.transpose(image, perm=[0,1,2])
# [x, y, channel] = [y, x, channel] ==> flipped = [1,0,2]
flipped = tf.transpose(image, perm=[1,0,2])
sess.run(vars)
result=sess.run(flipped)
plot.imshow(result)
plot.show()
