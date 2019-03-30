import numpy as np
x = np.arange(12)
y = np.resize(x,24) # creates a duplicate of x
print ('x = ',x)
print ('y = ',y)