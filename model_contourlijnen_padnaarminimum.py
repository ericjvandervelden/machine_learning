import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

x0, x1 = np.meshgrid( np.linspace(0,2, 11), np.linspace(0,4,21),)
f=(x0-1)**4+(x1-2)**4-7

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

# contourlijnen ,
plt.figure()
plt.gca().set_aspect('equal')
c= plt.contour(x0, x1, f,cmap=plt.cm.brg)
plt.clabel(c, fmt='%1.1f', inline_spacing=40,fontsize=10)

# pad naar minimum  ,
x=np.zeros((10,2))
for i in range(0,10-1):
  gradf=np.array([4*(x[i,0]-1)**3,4*(x[i,1]-2)**3])
  H=np.block([[4*3*(x[i,0]-1)**2,0],[0,4*3*(x[i,1]-2)**2]])
  x[i+1]=x[i]-np.linalg.inv(H).dot(gradf)
  display(x[i+1])
  plt.scatter(x[:,0],x[:,1],)
# or  ,
# plt.plot(x[:,0],x[:,1],)


plt.show()

