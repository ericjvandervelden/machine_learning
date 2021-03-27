import sys
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las

if len(sys.argv)!=6:
    print("Use: cg.py <herm pos def matrix> <target> <x0> <#iterations> <tolerance>\n",file=sys.stderr) 
    sys.exit(1) 

A=np.array(sys.argv[1].split(',')).astype(int)
m=np.sqrt(A.shape[0]).astype(int)
A=A.reshape(-1,m)
k=np.array(sys.argv[2].split(',')).astype(int)
n=int(sys.argv[4])
x=np.zeros((m,n))
x[:,0]=np.array(sys.argv[3].split(',')).astype(int)
p=np.zeros((m,n))
r=np.zeros((m,n))
a=np.zeros(n)
b=np.zeros(n)
tol=float(sys.argv[5])

print("x[:,0]:",x[:,0])
p[:,0]=r[:,0]=k-A@x[:,0]
print("p[:,0]:",p[:,0])
print("r[:,0]:",r[:,0])
for i in np.arange(0,n-1):
    print("ronde %d",i)
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
    
    
    








#A=np.array([2,-1,-1,2]).reshape(-1,2)
#lu,piv=la.lu_factor(A)
#x=la.lu_solve((lu,piv),grad)
