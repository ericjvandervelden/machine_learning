import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()





phi=iris['data'][:,int(sys.argv[2])]    # (150,) # geef features op,
Phi=np.c_[np.ones((phi.shape[0],1)),phi] #(150,#features+1)

t=iris.target
T=np.zeros([len(t),3])
for k in np.arange(3):
    T[:,k]=(t==k).astype(int)
# of    ,
#T=np.zeros([3,len(t)])
#for k in np.arange(3):
#    T[k,:]=(t==k).astype(int)

W=np.zeros((Phi.shape[1],np.unique(t).shape[0])) # #features +1, #classes  ,

#W=(w0,w1,w2), wi staande vectoren, dim=#features+1 ,
#T=(010
#   100
#   001
#   100
#   ...
# matrix ipv t=(1,0,2,0,...) die de classes geeft waarin elke waarneming zit    ,
# W@T.T=(w1,w0,w2,w0,...) # (150,#samples)
np.diag(Phi@W@T.T)
Phi@W       # W=(w0,w1,w2) # (#features+1,#classes)

np.exp(np.diag(Phi@W@T.T))/(np.exp(Phi@W)@np.ones(np.unique(t).shape[0]))

w=np.zeros(2) # (2,)
t=(iris['target']==int(sys.argv[1])).astype(int) # (150,)
p1=sp.special.expit(Phi@w)     # p1|phi # (150,)
grad=Phi.T @ (p1-t) # (2,)

for i in range(0,50):
    p1=sp.special.expit(Phi@w)     # p1|phi  (150,)
    grad=Phi.T @ (p1-t) # (2,)
    R=np.diag(p1*(1-p1))    # (150,150)
    H=Phi.T @ R @ Phi
    w=w-la.lu_solve(la.lu_factor(H),grad)
    display(w)

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

plt.figure()
plt.plot(phi[t==0],t[t==0],'gs')
plt.plot(phi[t==1],t[t==1],'b^')

grid=np.linspace(min(phi),max(phi),101)
Phi_grid=np.c_[np.ones(101,dtype='int'),grid]
t_grid=sp.special.expit(Phi_grid @ w)
plt.plot(grid,t_grid)

plt.show()

