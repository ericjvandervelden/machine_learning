import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=2:
    print("Use: og_reg_multiclass_self_202103.py <#iteraties>\n",file=sys.stderr) 
    sys.exit(1) 

# NF=#features
# NC=#classes
# NI=#iteraties
# NS=#samples

NI=int(sys.argv[1])

features=np.array([0])
NF=len(features)
my_data=np.array([-2,2]).reshape(-1,1)   # ipv iris.data
NS=my_data.shape[0]
phi=my_data[:,features]  # phi==my_data    
Phi=np.c_[np.ones(NS),phi] 

NC=2

# We willen maken (w1,w0,w2,w0,...) # NS in volgorde zoals de waargenomen classes zijn   ,
#W=(w0,w1,w2), vert. dim=#features+1 , #classes=3   ,
# W@T.T=(w1,w0,w2,w0,...) # (150,#samples)  , waar T:
#T=(010
#   100
#   001
#   100
#   ...
# matrix ipv t=(1,0,2,0,...) die de classes geeft waarin elke waarneming zit    ,

t=np.array([0,1])       # classes van samples   ,
T=np.zeros([NS,NC],dtype=int)
for k in np.arange(NC):
    T[:,k]=t==k

W=np.zeros((NF+1,NC)) 
for i in np.arange(0,np.minimum(NF+1,NC)):
    W[i,i]=1

print("T:\n")
print(T)
print("W:\n")
print(W)

for i in np.arange(0,NI):

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
    y=np.exp(np.diag(Phi @ W @ T.T))/( np.exp(Phi @ W) @ np.ones(NC) )  # NF+1 vector   , 

#
    np.ones(NS)@np.exp(Phi@W)
# deel iedere column door som van de column: 
    PcGs=np.exp(Phi@W)/( np.ones(NS) @ np.exp(Phi@W) )   # NS x NC, elke term= (4.104) = (c|s)
    print("PcGs:\n")
    print(PcGs)
# #classes grads op een rij; Bishop (4.109)
#Grads=Phi.T @ ( np.exp(Phi@W)/( np.ones(NS) @ np.exp(Phi@W) ) - T )
    Grad=Phi.T @ ( PcGs - T )   # NF+1 x NC
#array([[ -49.        ,  -49.        ,  -49.        ],
#       [ -11.00133333,  -65.10133333, -100.10133333]])

# hessian   ,
    D=np.array([]).reshape(0,NS*NC)
    for c in np.arange(0,NC):
        D=np.r_[D,np.kron(np.ones(NC),np.diag(PcGs[:,c]))]
    R=(np.eye(NS*NC)-D)*D.T  # (NS x NS)*NC
    Phi_NC=np.kron(np.eye(NC),Phi) # (NS x NF+1)*NC
    H=Phi_NC.T @ R @ Phi_NC # (NF+1)*NC x (NF+1)*NC
    grad=Grad.reshape(-1,1,order='f') # NF+1 * NC vector ; grad to wj is zo groot als een phi = NF+1
    w=W.reshape(-1,1,order='f') # of W.T.reshape(-1,1) # NF+1 * NC vector   ,
    w=w-la.lu_solve(la.lu_factor(H),grad) # grad naar wj is net zo groot als wj = NF+1
    W=w.reshape(-1,NC,order='f')
    print("W:\n" )
    print(W )







# len phi = len w  = #features +1
# er zijn #classes w's  ,

# bij #features+1=2 , 
#In [781]: Phi.T @ Phi
#Out[781]: 
#array([[150. , 179.8],
#       [179.8, 302.3]])
# maar dit zegt niets over het #classes ,
# onze matrix is #classes*(#features+1) groot,  






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
