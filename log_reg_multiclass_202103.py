import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=4:
    print("Use: og_reg_multiclass_20102.py <features> <#classes> <#iteraties>\n",file=sys.stderr) 
    sys.exit(1) 

# NF=#features
# NC=#classes
# NI=#iteraties
# NS=#samples
features=np.array(sys.argv[1].split(',')).astype(int)
NF=len(features)
NC=int(sys.argv[2])
NI=int(sys.argv[3])

# iris=sk.datasets.load_iris() # does not work TODO
iris=datasets.load_iris()
NS=iris.data.shape[0]

# we gaan nu uit van 1 feature, de laatste  ,
# roep script aan met #features=4, #classes=3

phi=iris.data[:,features]    # (NS,) # samples van laatste feature 
Phi=np.c_[np.ones(NS),phi] #(NS,#features+1)
#Phi=np.c_[np.ones((NS,1)),phi] # niet nodig (NS,1) ipv NS  ,

# We willen maken (w1,w0,w2,w0,...) # NS in volgorde zoals de waargenomen classes zijn   ,
#W=(w0,w1,w2), vert. dim=#features+1 , #classes=3   ,
# W@T.T=(w1,w0,w2,w0,...) # (150,#samples)  , waar T:
#T=(010
#   100
#   001
#   100
#   ...
# matrix ipv t=(1,0,2,0,...) die de classes geeft waarin elke waarneming zit    ,

t=iris.target
T=np.zeros([NS,NC],dtype=int)
for k in np.arange(NC):
    T[:,k]=t==k
# of    ,
#T=np.zeros([3,len(t)])
#for k in np.arange(3):
#    T[k,:]=(t==k).astype(int)

W=np.zeros((NF+1,NC)) # (#features +1, #classes) 
                                                    # w's die naast elkaar rechtop staan  ,
W[0,:]=1

#W=(w0,w1,w2), wi staande vectoren, vert. dim=#features+1 , #classes=3 hier,    
#T=(010
#   100
#   001
#   100
#   ...
# matrix ipv t=(1,0,2,0,...) die de classes geeft waarin elke waarneming zit    ,
# W@T.T=(w1,w0,w2,w0,...) # (150,#samples)
np.diag(Phi@W@T.T)
Phi@W       # W=(w0,w1,w2) # (#features+1,#classes)

# Bishop (p.209) (4.104)    , maar voor iedere waarneming phi met z'n class (supervised learning),
# hier kun je de lh mee maken   , TODO
y=np.exp(np.diag(Phi@W@T.T))/( np.exp(Phi@W)@np.ones(NC) )

#
np.ones(NS)@np.exp(Phi@W)
# deel iedere column door som van de column: 
PcGs=np.exp(Phi@W)/( np.ones(NS) @ np.exp(Phi@W) )   # NS x NC, elke term= (4.104) = (c|s)
# #classes grads op een rij; Bishop (4.109)
#Grads=Phi.T @ ( np.exp(Phi@W)/( np.ones(NS) @ np.exp(Phi@W) ) - T )
Grads=Phi.T @ ( PcGs - T )   # NF+1 x NC
#array([[ -49.        ,  -49.        ,  -49.        ],
#       [ -11.00133333,  -65.10133333, -100.10133333]])

# hessian   ,
D=np.array([]).reshape(0,NS*NC)
for c in np.arange(0,NC):
    D=np.r_[D,np.kron(np.ones(NC),np.diag(PcGs[:,c]))]
R=(np.eye(NS*NC)-D)*D.T  # NS*NC x NS*NC
H=Phi.T @ R @ Phi    # TODO
w=w-la.lu_solve(la.lu_factor(H),grad)







# len phi = len w  = #features +1
# er zijn #classes w's  ,

# bij #features+1=2 , 
#In [781]: Phi.T @ Phi
#Out[781]: 
#array([[150. , 179.8],
#       [179.8, 302.3]])
# maar dit zegt niets over het #classes ,
# onze matrix is #classes*(#features+1) groot,  





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

