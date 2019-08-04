import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

x=iris['data'][:,[2,3]]
X=np.c_[np.ones((len(x),1)),x]
theta=np.array([0,0,0]).reshape(-1,1)
y=(iris['target']==2).reshape(-1,1).astype(int)

from scipy.special import expit

for i in range(0,10):
  z=X.dot(theta)
  a=expit(z)
  grad=X.T.dot(a-y)
  D=np.diag((a*(1-a)).ravel())
  H=X.T.dot(D).dot(X)
  theta=theta-np.linalg.inv(H).dot(grad)

display(theta)

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

plt.figure()
plt.plot(x[(y==0).ravel(),0],x[(y==0).ravel(),1],'gs')
plt.plot(x[(y==1).ravel(),0],x[(y==1).ravel(),1],'b^')

i=np.linspace(min(x[:,0]),max(x[:,0]),60)
j=np.linspace(min(x[:,1]),max(x[:,1]),25)
I,J=np.meshgrid(i,j)
G=np.c_[np.ones(len(I.ravel())),I.ravel(),J.ravel()]

V=expit(G.dot(theta)).reshape(I.shape)
contours=plt.contour(I,J,V,cmap=plt.cm.brg)
plt.clabel(contours,inline=True)

plt.show()

