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

    print("x[:,0]:",x[:,0])
    p[:,0]=r[:,0]=k-A@x[:,0]
    print("p[:,0]:",p[:,0])
    print("r[:,0]:",r[:,0])
    for i in np.arange(0,n-1):
        print("ronde:",i)
        a[i]=(r[:,i] @ r[:,i])/( p[:,i] @ A @ p[:,i] )
        print("a[i]:",a[i])
        x[:,i+1]=x[:,i]+a[i]*p[:,i]
        print("x[:,i+1]:",x[:,i+1])
        sol=x[:,i+1]
        if la.norm(k-A@x[:,i+1])<tol:#*la.norm(k):
            print("break")
            break  
        r[:,i+1]=r[:,i]-a[i]*A @ p[:,i]
        print("r[:,i+1]:",r[:,i+1])
        b[i]=( r[:,i+1] @ r[:,i+1] )/( r[:,i] @ r[:,i] )
        print("b[i]:",b[i])
        p[:,i+1]=r[:,i+1] + b[i]*p[:,i]
        print("p[:,i+1]:",p[:,i+1])
    print("i:",i)
    print("sol:",sol)

cg("2,1,0,1,2,1,0,1,2","3,4,3","0,0,0","5","1e-5")	    

#A=np.array([2,-1,-1,2]).reshape(-1,2)
#lu,piv=la.lu_factor(A)
#x=la.lu_solve((lu,piv),grad)
