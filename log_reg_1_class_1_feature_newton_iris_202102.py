import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=4:
    print("Use: log_reg_1_class_1_feature_newton_iris_202102.py <features> <target> <#iterations>\n",file=sys.stderr)
    sys.exit(1)

def sgm(x):
    return 1/(1+np.exp(-x))


# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()

phi=iris['data'][:,int(sys.argv[1])]    # (150,)
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (150,2)
w=np.zeros(2) # (2,)
t=(iris['target']==int(sys.argv[2])).astype(int) # (150,)
niter=int(sys.argv[3])

for i in np.arange(0,niter):
    #p1=sp.special.expit(Phi@w)     # p1|phi  (150,)
    p1=sgm(Phi@w)     # p1|phi  (150,)

    cost=np.ones(phi.shape[0])@(t*-np.log(p1)+(1-t)*-np.log(1-p1))
    print("cost=",cost)

    grad=Phi.T @ (p1-t) # (2,)
    print("grad=",grad)
    R=np.diag(p1*(1-p1))    # (150,150)
    H=Phi.T @ R @ Phi
    w=w-la.lu_solve(la.lu_factor(H),grad)
    print("w=",w)

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
