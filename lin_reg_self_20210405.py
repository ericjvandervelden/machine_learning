import numpy as np
import numpy.linalg as la

# lin regressie met Phi+    ,
# Phi=
#[1,-1
# 1,0
# 1,1]
# Phi+=
#[2,1,-1 /3
#-1,1,2 /3]
# t ligt in cols Phi    , dus Phi @ w = t
# t2 ligt in niet in cols Phi, dus Phi @ w2 != t2, maar is de loodrechte proj op cols Phi van t2 , 
# antwoord: w=(1,1),w2=(4/3,1/2)
phi=np.array([-1,0,1])
Phi=np.c_[np.ones(phi.shape),phi]
t=np.array([0,1,2])
t2=np.array([1,1,2])
inner=Phi.T @ Phi
inner_inv=la.inv(inner)
Phi_plus=inner_inv @ Phi.T
w=Phi_plus @ t
w2=Phi_plus @ t2
eq=np.allclose(Phi @ w, t))
eq2=np.isclose( Phi.T @ (t2-Phi@w2 ),0 )

# we bereken Phi+  met la.pinv, via de svd, en met la.inv(Phi.T @ Phi) @ Phi.T  ,
Phi=np.array([[1,1,0],[0,1,1]]).T
Phi_plus=la.pinv(Phi)
inner=Phi.T @ Phi
inner_inv=la.inv(inner)
Phi_plus2=inner_inv @ Phi.T
# la.svd geeft in V en UT -'en  ,
V,s,UT=la.svd(Phi,full_matrices=False)
U=UT.T
Phi_plus3=U @ np.diag(1/s) @ V.T
np.allclose(Phi_plus,Phi_plus2)
np.allclose(Phi_plus,Phi_plus3)

Phi=np.array([[1,1,0],[0,1,1]]).T
w=np.zeros(Phi.shape[1]);w[0]=1
t=np.array([1,2,1])
alpha=.01
niter=10
for i in np.arange(0,niter):
    w=w-alpha*Phi.T @ (Phi @ w - t)
    c=np.square(la.norm(Phi @ w - t))
    print("w=",w,", c=",c)




