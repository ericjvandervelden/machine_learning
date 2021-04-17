import sys
import numpy as np
from scipy import special 
import scipy.linalg as la

if len(sys.argv)!=4:
    print("Use: log_reg_1_class_features_newton_self_202102.py  <samples> <classes> <#iterations>\n",file=sys.stderr) 
    sys.exit(1) 

phi=np.array(sys.argv[1].split(',')).astype(int)#.reshape(-1,1) #(2,)
t=np.array(sys.argv[2].split(',')).astype(int)   #(2,)
n=int(sys.argv[3])
Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (2,2)
w=np.zeros(2) # (2,)
for i in range(0,n):
    sgm=special.expit(Phi@w)     # p(1|phi)=sgm  (2,)
    print("sgm")
    print(sgm)
    grad=Phi.T @ (sgm-t) # (2,)
    print("grad")
    print(grad)
    R=np.diag(sgm*(1-sgm))    # (2,2)
    H=Phi.T @ R @ Phi
    print("H")
    print(H)
    itm=la.lu_solve(la.lu_factor(H),grad)
    print("itm")
    print(itm)
    w=w-itm
    #w=w-la.lu_solve(la.lu_factor(H),grad)
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
#t_grid=special.expit(Phi_grid @ w)
#plt.plot(grid,t_grid)
#
#plt.show()
#
