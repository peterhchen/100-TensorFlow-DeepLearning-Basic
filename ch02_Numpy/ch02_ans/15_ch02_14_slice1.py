import numpy as np
a = np.arange(8)
print('a:  ',a)
d1 = np.asanyarray(a[:1:])
print('d1: ',d1)
d2 = np.asanyarray(a[:2:])
print('d2: ',d2)
d3 = np.asanyarray(a[:3:])
print('d3: ',d3)
range1 = 4
d4 = np.asanyarray(a[:range1:])
print('d4: ',d4)