import numpy as np
from scipy import special 
import scipy.linalg as la

phi=np.array([-1,1])     # (2,)
t=np.array([0,1]) # (2,)

Phi=np.c_[np.ones(len(phi),dtype='int'),phi] # (2,2)
w=np.zeros(2) # (2,)
for i in range(0,100):
    sgm=special.expit(Phi@w)     # p(1|phi)=sgm  (2,)
    grad=Phi.T @ (sgm-t) # (2,)
    R=np.diag(sgm*(1-sgm))    # (2,2)
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
t_grid=special.expit(Phi_grid @ w)
plt.plot(grid,t_grid)

plt.show()

