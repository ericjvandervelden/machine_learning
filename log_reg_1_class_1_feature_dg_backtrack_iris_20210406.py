import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=8:
    print("Use: log_reg_1_class_1_feature_dg_backtrack_iris_20210406.py <features> <target> <#iterations gd> <absolute tolerance> <lift coeff c> <gamma g> <#iterations bt>\n",file=sys.stderr) 
    sys.exit(1) 

def sgm(x):
    return 1/(1+np.exp(-x))

# Phi, t globals, constants , 
def fcost(w):
    p1=sgm(Phi@w)     # p1|phi # (150,)
    return np.ones(Phi.shape[0])@(t*-np.log(p1)+(1-t)*-np.log(1-p1))

def fgrad(w):
    p1=sgm(Phi@w)     # p1|phi # (150,)
    return Phi.T @ (p1-t) # (2,)

# c is lift coeff   , waarmee je de raaklijn omhoog lift    , global, const ,
def bt(w,grad):
    n=0
    while n<niter_bt:
        s=g**n
        #print(t,p(0+t*d),p(0)+c*l(0)*t*d)
        #plt.plot(t*d,p(0)+c*l(0)*t*d,'gs')
        if fcost(w+s*-grad/la.norm(grad))<=fcost(w)+c*s*-la.norm(grad):
            break
        n=n+1
    print("n_bt=",n)
    return s 


# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()
phi=iris['data'][:,int(sys.argv[1])]    # (150,) # phi = features   ,
#print("phi=",phi)
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (150,2)
#print("Phi=",Phi)
t=(iris['target']==int(sys.argv[2])).astype(int) # (150,)
#print("t=",t)
niter_gd=int(sys.argv[3])
atol=float(sys.argv[4])
c=float(sys.argv[5])
g=float(sys.argv[6])
niter_bt=int(sys.argv[7])

w=np.zeros(2) # (2,)
cost_prev=fcost(w)
#print("cost_prev=",cost_prev)
for i in range(0,niter_gd):
    grad=fgrad(w)
    s=bt(w,grad)
    print("s=",s)
    #print("grad=",grad)
    w=w+s*-grad/la.norm(grad)
    print("w=",w)
    cost=fcost(w)
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
