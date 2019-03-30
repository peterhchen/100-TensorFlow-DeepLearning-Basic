import matplotlib.pyplot as plt #cross-entropy.py
import numpy as np

y = 0.5
def lossCE(p,y):
  loss = -(y*np.log(p)+(1-y)*np.log(1-p))
  return loss

probabilities = np.arange(1e-3,0.999,1e-3)
loss = lossCE(probabilities,y)
plt.plot(probabilities,loss)
plt.title('Cross Entropy Loss')
plt.show()