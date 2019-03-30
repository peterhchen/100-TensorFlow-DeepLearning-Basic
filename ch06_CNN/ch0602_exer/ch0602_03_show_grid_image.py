import matplotlib.pyplot as plt # show-grid-images.py
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
batch_xs, batch_ys = mnist.train.next_batch(100)

rows = 3
cols = 5
count = rows*cols
fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(count):
  img = batch_xs[i].reshape(28, 28)
  ax[i].imshow(img, interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()

# plt.savefig('mnist_figures.png', dpi=300)
plt.show()