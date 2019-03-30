import numpy as np
import matplotlib.pyplot as plt

x = [2,3,4,5,7,9,13,15,17]
plt.plot(x) # OR: plt.plot(x, ‘ro-’) or bo
plt.ylabel('Height')
plt.xlabel('Weight')
plt.show()