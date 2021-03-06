import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=6:
    print("Use: log_reg_1_class_1_feature_dg_iris_20210406.py <features> <target> <alpha> <#iterations> <absolute tolerance>\n",file=sys.stderr) 
    sys.exit(1) 

def sgm(x):
    return 1/(1+np.exp(-x))


# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()
phi=iris['data'][:,int(sys.argv[1])]    # (150,) # phi = features   ,
#print("phi=",phi)
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (150,2)
#print("Phi=",Phi)
t=(iris['target']==int(sys.argv[2])).astype(int) # (150,)
#print("t=",t)
alpha=float(sys.argv[3])
niter=int(sys.argv[4])
atol=float(sys.argv[5])

w=np.zeros(2) # (2,)
p1=sgm(Phi@w)     # p1|phi # (150,)
#print("p1=",p1)
cost_prev=np.ones(phi.shape[0])@(t*-np.log(p1)+(1-t)*-np.log(1-p1))
#print("cost_prev=",cost_prev)
for i in range(0,niter):
    grad=Phi.T @ (p1-t) # (2,)
    print("grad=",grad)
    w=w-alpha*grad
    print("w=",w)
    p1=sgm(Phi@w)     # p1|phi # (150,)
    #print("p1=",p1)
    cost=np.ones(phi.shape[0])@(t*-np.log(p1)+(1-t)*-np.log(1-p1))
    #print("cost=",cost)
    if(cost_prev<cost):
        print("verkeerde richting; stijgende cost")
    if(np.abs(cost_prev-cost)<atol):
        print("klaar")
        print("w=",w,", i=",i,", cost_prev=",cost_prev,", cost=",cost)
        break
    cost_prev=cost
    


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
