import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

x=iris['data'][:,[3]]
X=np.c_[np.ones((len(x),1)),x]
theta=np.array([0,0]).reshape(-1,1)
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
plt.plot(x[(y==0).ravel()],y[(y==0).ravel()],'gs')
plt.plot(x[(y==1).ravel()],y[(y==1).ravel()],'b^')

i=np.linspace(min(x.ravel()),max(x.ravel()),101)
I=np.c_[np.ones(101),i]
v=expit(I.dot(theta))
plt.plot(i,v)

plt.show()

