import sys
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las

if len(sys.argv)!=6:
    print("Use: cg.py <size matrix> <matrix> <target> <x0> <#iteraties>\n",file=sys.stderr) 
    sys.exit(1) 

m=int(sys.argv[1])
A=np.array(sys.argv[2].split(',')).astype(int).reshape(-1,m)
k=np.array(sys.argv[3].split(',')).astype(int)
n=int(sys.argv[4])
x=np.zeros((m,n))
x[:,0]=np.array(sys.argv[5].split(',')).astype(int)
p=np.zeros((m,n))
r=np.zeros((m,n))
a=np.zeros(n)
b=np.zeros(n)

p[:,0]=r[:,0]=k-A@x[:,0]
for i in np.arange(0,n-2):
    a[i]=(r[:,i] @ r[:,i])/( p[:,i] @ A @ p[:,i] )
    x[:,i+1]=x[:,i]+a[i]*p[:,i]
    r[:,i+1]=r[:,i]-a[i]*A @ p[:,i]
    b[i]=( r[:,i+1] @ r[:,i+1] )/( r[:,i] @ r[:,i] )
    p[:,i+1]=r[:,i+1] + b[i]*p[:,i]

print(x[:,n-1])
    
    
    








#A=np.array([2,-1,-1,2]).reshape(-1,2)
#lu,piv=la.lu_factor(A)
#x=la.lu_solve((lu,piv),grad)
