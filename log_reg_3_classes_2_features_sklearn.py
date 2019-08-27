import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

from sklearn.linear_model import LogisticRegression

X=iris['data'][:,[2,3]]
lg=LogisticRegression(C=10**10)
y=iris['target']
lg.fit(X,y)

print("lg.intercept_")
print(lg.intercept_)

print("lg.coef_")
print(lg.coef_)



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
