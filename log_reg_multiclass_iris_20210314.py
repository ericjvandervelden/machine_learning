import sys
import numpy as np
from sklearn import datasets
# import sklearn as sk
import scipy as sp
import scipy.linalg as la

if len(sys.argv)!=3:
    print("Use: og_reg_multiclass_20102.py <features> <#iteraties>\n",file=sys.stderr)
    sys.exit(1)


# NF=#features
# NC=#classes
# NI=#iteraties
# NS=#samples

features=np.array(sys.argv[1].split(',')).astype(int)
NF=features.shape[0]
NI=int(sys.argv[2])

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

t=iris.target   # classes van samples   ,
NC=np.unique(t).shape[0]           # aanpassing 1
T=np.zeros([NS,NC],dtype=int)
for k in np.arange(NC):
    T[:,k]=t==k         # set col T

W=np.zeros((NF+1,NC)) 
for i in np.arange(0,np.minimum(NF+1,NC)):
    W[i,i]=1

print("T:\n")
print(T.shape)
print(T)
print("W:\n")
print(W.shape)
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
    Int=np.exp(Phi@W)   # NS x NC               # aanpassing 2
    print("Int:\n" )
    print(Int.shape)
    print(( np.ones((NS,1)) ).shape)
    Pcs=Int/( Int @ np.ones((NC,1)) )   # NS x NC, elke term= (4.104) = (c|s) # aanpassing 2
                                    # comment deel door de som over alle classes    ,
    print("Pcs:\n")
    print(Pcs.shape)
    print(Pcs)
# #classes grads op een rij; Bishop (4.109)
#Grads=Phi.T @ ( np.exp(Phi@W)/( np.ones(NS) @ np.exp(Phi@W) ) - T )
    Grad=Phi.T @ ( Pcs - T )   # NF+1 x NC
                                    # comment sommeer over alle samples,    
    print("Grad:\n")
    print(Grad.shape)
    print(Grad)
#array([[ -49.        ,  -49.        ,  -49.        ],
#       [ -11.00133333,  -65.10133333, -100.10133333]])

    Int0=np.block([np.diag(Pcs[:,j]) for j in range(NC)])
    Int1=Int0.T @ -Int0
    Int2=np.diag(np.block([Pcs[:,j] for j in range(NC)]))
    Int=Int2+Int1
    print("Int:\n")
    print(Int.shape)
    print(Int)
    Phi_NC_diag=np.kron(np.eye(NC),Phi)
    Hessian=Phi_NC_diag.T @ Int @ Phi_NC_diag
    print("Hessian:\n" )
    print(Hessian.shape)
    print(Hessian)
    grad=Grad.reshape(-1,1,order='f') # NF+1 * NC vector ; grad to wj is zo groot als een phi = NF+1
    print("Grad:\n")
    print(Grad.shape)
    print(Grad)
    w=W.reshape(-1,1,order='f') # of W.T.reshape(-1,1) # NF+1 * NC vector   ,
    print("w:\n")
    print(w.shape)
    print(w)
    lu, piv=la.lu_factor(Hessian)   # later rm  ,
    print("lu:\n")
    print(lu.shape)
    print(lu)
    print("piv:\n")
    print(piv.shape)
    print(piv)
    w=w-la.lu_solve(la.lu_factor(Hessian),grad) # grad naar wj is net zo groot als wj = NF+1
    W=w.reshape(-1,NC,order='f')
    print("W:\n")
    print(W.shape)
    print(W)

    
 

## hessian   ,
#    D=np.array([]).reshape(0,NS*NC)
#    for c in np.arange(0,NC):
#        D=np.r_[D,np.kron(np.ones(NC),np.diag(PcGs[:,c]))]
#    R=(np.eye(NS*NC)-D)*D.T  # (NS x NS)*NC
#    Phi_NC=np.kron(np.eye(NC),Phi) # (NS x NF+1)*NC
#    H=Phi_NC.T @ R @ Phi_NC # (NF+1)*NC x (NF+1)*NC
#    grad=Grad.reshape(-1,1,order='f') # NF+1 * NC vector ; grad to wj is zo groot als een phi = NF+1
#    w=W.reshape(-1,1,order='f') # of W.T.reshape(-1,1) # NF+1 * NC vector   ,
#    w=w-la.lu_solve(la.lu_factor(H),grad) # grad naar wj is net zo groot als wj = NF+1
#    W=w.reshape(-1,NC,order='f')
#    print("W:\n" )
#    print(W )
#
#
#
#
#
#
#
## len phi = len w  = #features +1
## er zijn #classes w's  ,
#
## bij #features+1=2 , 
##In [781]: Phi.T @ Phi
##Out[781]: 
##array([[150. , 179.8],
##       [179.8, 302.3]])
## maar dit zegt niets over het #classes ,
## onze matrix is #classes*(#features+1) groot,  
#
#
#
#
#
#
##import matplotlib.pyplot as plt
##import seaborn
##seaborn.set()
##
##plt.figure()
##plt.plot(phi[t==0],t[t==0],'gs')
##plt.plot(phi[t==1],t[t==1],'b^')
##
##grid=np.linspace(min(phi),max(phi),101)
##Phi_grid=np.c_[np.ones(101,dtype='int'),grid]
##t_grid=sp.special.expit(Phi_grid @ w)
##plt.plot(grid,t_grid)
##
##plt.show()
##
