import numpy as np
x = np.arange(12)
print (x.reshape(3,4))    # 12 elements
print (x.reshape(6,2))    # 12 elements
print (x.reshape(6,4))    # Error