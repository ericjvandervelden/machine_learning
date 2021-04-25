import sys
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn

if len(sys.argv)!=4:
    print("Use: log_reg_1_class_features_self_geron_20210315.py samples classes tol\n",file=sys.stderr) 
    sys.exit(1) 

phi=np.array(sys.argv[1].split(',')).astype(int).reshape(-1,1)
t=np.array(sys.argv[2].split(',')).astype(int)
tol_=float(sys.argv[3])
lg=LogisticRegression(solver="newton-cg",C=10**10,verbose=1,tol=tol_) 

lg.fit(phi,t)

print("intercept_")
print(lg.intercept_)
print("coef_")
print(lg.coef_)

#x0, x1 = np.meshgrid(
#        np.linspace(0,8, 500).reshape(-1, 1),
#        np.linspace(0,3.5, 200).reshape(-1, 1),
#)
#X_new = np.c_[x0.ravel(), x1.ravel()]
#y_proba = lg.predict_proba(X_new)
#y_predict = lg.predict(X_new)
#
#
#prb0=y_proba[:,0].reshape(x0.shape)
#prb1=y_proba[:,1].reshape(x0.shape)
#prb2=y_proba[:,2].reshape(x0.shape)
#est=y_predict.reshape(x0.shape)
#
#
#seaborn.set()
#
## print training data ,
#plt.figure()
#plt.plot(X[y==2,0],X[y==2,1],"g^",label="Iris-Virginia")
#plt.plot(X[y==1,0],X[y==1,1],"bs",label="Iris-Versicolor")
#plt.plot(X[y==0,0],X[y==0,1],"yo",label="Iris-Setosa")
#
#from matplotlib.colors import ListedColormap
#custom_cmap=ListedColormap(['#fafab0','#9898ff','#a0faa0'])
#plt.contourf(x0,x1,est,cmap=custom_cmap)
#
#countour1=plt.contour(x0,x1,prb1,cmap=plt.cm.brg)
#countour0=plt.contour(x0,x1,prb0,cmap=plt.cm.brg)
#countour2=plt.contour(x0,x1,prb2,cmap=plt.cm.brg)
#
#plt.clabel(countour1,inline=1,fontsize=12)
#plt.clabel(countour0,inline=1,fontsize=12)
#plt.clabel(countour0,inline=1,fontsize=12)
#
#plt.show()
#
