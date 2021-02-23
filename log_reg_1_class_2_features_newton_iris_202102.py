import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()

phi=iris['data'][:,[int(sys.argv[2]),int(sys.argv[3])]]    # (150,2)
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (150,3)
w=np.zeros(3) # (3,)
t=(iris['target']==int(sys.argv[1])).astype(int) # (150,)
p1=sp.special.expit(Phi@w)     # p1|phi # (150,)
grad=Phi.T @ (p1-t) # (3,)

for i in range(0,10):
    p1=sp.special.expit(Phi@w)     # p1|phi  (150,)
    grad=Phi.T @ (p1-t) # (3,)
    R=np.diag(p1*(1-p1))    # (150,150)
    H=Phi.T @ R @ Phi
    w=w-la.lu_solve(la.lu_factor(H),grad)
    display(w)

#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set()
#
#plt.figure()
#plt.plot(phi[t==0],t[t==0],'gs')
#plt.plot(phi[t==1],t[t==1],'b^')
#
#grid=np.linspace(min(phi),max(phi),101)
#Phi_grid=np.c_[np.ones(101,dtype='int'),grid]
#t_grid=sp.special.expit(Phi_grid @ w)
#plt.plot(grid,t_grid)
#
#plt.show()
#
