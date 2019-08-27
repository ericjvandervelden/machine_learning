import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

from scipy.special import expit

def compute_params_for_class(c):
  X=iris['data'][:,[2,3]]
  X=np.c_[np.ones((X.shape[0],1)),X]
  theta=np.array([0,0,0]).reshape(-1,1)
  y=(iris['target']==c).reshape(-1,1).astype(int)
# range(0,80): class 1 en 2 OK  , class 0 -> blows up   ,
  for i in range(0,10):
    z=X.dot(theta)
    a=expit(z)
    grad=X.T.dot(a-y)
    D=np.diag((a*(1-a)).ravel())
    H=X.T.dot(D).dot(X)
    theta=theta-np.linalg.inv(H).dot(grad)
  return theta

print("class 0")
display(compute_params_for_class(0))
print("class 1")
display(compute_params_for_class(1))
print("class 2")
display(compute_params_for_class(2))


#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set()
#
#plt.figure()
#plt.plot(x[(y==0).ravel(),0],x[(y==0).ravel(),1],'gs')
#plt.plot(x[(y==1).ravel(),0],x[(y==1).ravel(),1],'b^')
#
#i=np.linspace(min(x[:,0]),max(x[:,0]),60)
#j=np.linspace(min(x[:,1]),max(x[:,1]),25)
#I,J=np.meshgrid(i,j)
#G=np.c_[np.ones(len(I.ravel())),I.ravel(),J.ravel()]
#
#V=expit(G.dot(theta)).reshape(I.shape)
#contours=plt.contour(I,J,V,cmap=plt.cm.brg)
#plt.clabel(contours,inline=True)
#
#plt.show()
#
