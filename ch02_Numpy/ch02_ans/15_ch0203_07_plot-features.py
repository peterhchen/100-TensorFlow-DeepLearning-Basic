import numpy as np # plot-features.py
import matplotlib.pyplot as plt

greyhounds = 500
labradors  = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
labs_height = 24 + 4 * np.random.randn(labradors)

plt.hist([grey_height, labs_height],stacked=True,color=['r','b'])
plt.show()