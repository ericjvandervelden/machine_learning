import sys
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las

def cg(A,k,x0,niter,tol):
    A=np.array(A.split(',')).astype(int)
    m=np.sqrt(A.shape[0]).astype(int)
    A=A.reshape(-1,m)
    k=np.array(k.split(',')).astype(int)
    n=int(niter)
    x=np.zeros((m,n))
    x[:,0]=np.array(x0.split(',')).astype(int)
    p=np.zeros((m,n))
    r=np.zeros((m,n))
    a=np.zeros(n)
    b=np.zeros(n)
    tol=float(tol)

    p[:,0]=r[:,0]=k-A@x[:,0]
    for i in np.arange(0,n-1):
        a[i]=(r[:,i] @ r[:,i])/( p[:,i] @ A @ p[:,i] )
        x[:,i+1]=x[:,i]+a[i]*p[:,i]
        sol=x[:,i+1]
        if la.norm(k-A@x[:,i+1])<tol:#*la.norm(k):
            break  
        r[:,i+1]=r[:,i]-a[i]*A @ p[:,i]
        b[i]=( r[:,i+1] @ r[:,i+1] )/( r[:,i] @ r[:,i] )
        p[:,i+1]=r[:,i+1] + b[i]*p[:,i]
    return sol
	    
sol=cg("2,1,0,1,2,1,0,1,2","3,4,3","0,0,0","5","1e-5")	    
print("sol:",sol)
	    
	
def cg_(A,k,x0,niter,tol):
    m=A.shape[0]
    n=niter
    x=np.zeros((m,n))
    x[:,0]=x0
    p=np.zeros((m,n))
    r=np.zeros((m,n))
    a=np.zeros(n)
    b=np.zeros(n)

    p[:,0]=r[:,0]=k-A@x[:,0]
    for i in np.arange(0,n-1):
        a[i]=(r[:,i] @ r[:,i])/( p[:,i] @ A @ p[:,i] )
        x[:,i+1]=x[:,i]+a[i]*p[:,i]
        sol=x[:,i+1]
        if la.norm(k-A@x[:,i+1])<tol:#*la.norm(k):
            break  
        r[:,i+1]=r[:,i]-a[i]*A @ p[:,i]
        b[i]=( r[:,i+1] @ r[:,i+1] )/( r[:,i] @ r[:,i] )
        p[:,i+1]=r[:,i+1] + b[i]*p[:,i]
    return sol

A=np.array([2,1,0,1,2,1,0,1,2]).reshape(-1,3)
k=np.array([3,4,3])
x0=np.zeros(3)
niter=50	
tol=1e-5
sol=cg_(A,k,x0,niter,tol)	    
print("sol:",sol)
	
	
	
	
	
#A=np.array([2,-1,-1,2]).reshape(-1,2)
#lu,piv=la.lu_factor(A)
#x=la.lu_solve((lu,piv),grad)
