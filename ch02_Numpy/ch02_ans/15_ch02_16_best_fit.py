import numpy as np
xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([1,2,3,4,5], dtype=np.float64)

def best_fit_slope(xs,ys):
  m = (((np.mean(xs)*np.mean(ys))-np.mean(xs*ys)) /
      ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
  return m 

m = best_fit_slope(xs,ys)
print('m:',m)
b = np.sum(ys) - m * np.sum(xs)
print('b:',b)
