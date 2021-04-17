import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=9:
    print("Use: log_reg_1_class_1_feature_newton_backtrack_wolfe_weak_self_202102.py <features> <target> <#iterations newton> <absolute tolerance> <lift coeff c1> <coeff c2> <gamma c> <#iterations bt>",file=sys.stderr)
    sys.exit(1)


# we deden  , 
# w=w-la.lu_solve(la.lu_factor(H),grad)
# newton=la.lu_solve(la.lu_factor(H),grad) is de richting in de w-ruimte; bij gd is dat -grad  , 
# coursera: w=w-alpha*grad  , je zou kunnen doen: w=w-alpha*newton  , 
# maar we gaan alpha met backtracking uitrekenen    ,

phi=np.array(sys.argv[1].split(',')).astype(int).reshape(-1,1) # (..,)
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (...,2)
t=np.array(sys.argv[2].split(',')).astype(int)
niter_newton=int(sys.argv[3])
atol=float(sys.argv[4])
c1=float(sys.argv[5])
c2=float(sys.argv[6])
g=float(sys.argv[7])
niter_bt=int(sys.argv[8])

def sgm(x):
    return 1/(1+np.exp(-x))

# Phi, t globals, constants , 
def fcost(w):
    p1=sgm(Phi@w)     # p1|phi # (150,)
    print("p1=",p1)
    #cost=np.ones(Phi.shape[0])@(t*-np.log(p1)+(1-t)*-np.log(1-p1))
    print("t=",t)
    #print("cost=",np.where(t>0,-np.log(p1),-np.log(1-p1)))
    #cost2=np.ones(Phi.shape[0])  @ np.where(t>0,-np.log(p1),-np.log(1-p1))
    cost3v=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        if t[i]>0:
            cost3v[i]=-np.log(p1[i])
        else:
            cost3v[i]=-np.log(1-p1[i])
    cost3=np.ones(Phi.shape[0])  @ cost3v
    print("cost3=",cost3)
    return cost3


def fgrad(w):
    p1=sgm(Phi@w)     # p1|phi # (150,)
    return Phi.T @ (p1-t) # (2,)

def fnewton(w):
    p1=sgm(Phi@w)     # p1|phi # (150,)
    #3grad=Phi.T @ (p1-t) # (2,)
    #print("grad=",grad)
    R=np.diag(p1*(1-p1))    # (150,150)
    H=Phi.T @ R @ Phi
    return -la.lu_solve(la.lu_factor(H),fgrad(w))


# c1 is lift coeff   , waarmee je de raaklijn omhoog lift    , global, const ,
# d = gd, of d = newton
def bt(w,d):
    n=0
    while n<niter_bt:
        s=g**n
        #print(t,p(0+t*d),p(0)+c*l(0)*t*d)
        #plt.plot(t*d,p(0)+c*l(0)*t*d,'gs')
        if fcost(w+s*d)<=fcost(w)+c1*fgrad(w).T @ (s*d):
            if np.abs(fgrad(w+s*d).T @ d) < c2*np.abs(fgrad(w).T @ d):  
                print("break" )
                break
        n=n+1
    print("n_bt=",n)
    return s


w=np.zeros(2) # (2,)
cost_prev=fcost(w)
for i in np.arange(0,niter_newton):
    #print("cost=",cost)
    print("w=",w)
    #grad=fgrad(w)
    #print("grad=",grad)
    d=fnewton(w)
    print("d=",d)
    s=bt(w,d)
    print("s=",s)
    w=w+s*d
    cost=fcost(w) 
    print("cost_prev=",cost_prev,", cost=",cost,", diff=",np.abs(cost_prev-cost))
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
