import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

X=iris['data'][:,[2,3]]
X=np.c_[np.ones((X.shape[0],1)),X]
t=iris['target']
t=t.reshape(-1,1)
#beta1=np.array([0,0,0])
#beta2=np.array([0,0,0])
beta1=np.array([-2,1,-3])
beta2=np.array([-45,5,10])
beta1=beta1.reshape(-1,1)
beta1=beta2.reshape(-1,1)

from scipy.special import expit

# = expit, maar deze vorm in hastie (120)
def f1(y1,y2):
    return np.exp(y1)/(np.exp(y1)+np.exp(y2)+1)
def f2(y1,y2):
    return np.exp(y2)/(np.exp(y1)+np.exp(y2)+1)

for i in range(0,10):
    z1=X.dot(beta1)
    z2=X.dot(beta2)
    p1gx=f1(z1,z2)
    grad1=X.T.dot((t==1).astype(int)-p1gx)
    p2gx=f2(z1,z2)
    grad2=X.T.dot((t==2).astype(int)-p2gx)

    D11=-np.diag((p1gx*(1-p1gx)).ravel())
    D12=-np.diag((p1gx*(-p2gx)).ravel())
    D21=-np.diag((p2gx*(-p1gx)).ravel())
    D22=-np.diag((p2gx*(1-p2gx)).ravel())
    H11=X.T.dot(D11).dot(X)
    H12=X.T.dot(D12).dot(X)
    H21=X.T.dot(D21).dot(X)
    H22=X.T.dot(D22).dot(X)
    H=np.block([[H11,H12],[H21,H22]])
    beta=np.r_[beta1,beta2]-np.linalg.inv(H).dot(np.r_[grad1,grad2])
    beta1=beta[:3]
    beta2=beta[-3:]
    
display(beta)

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
