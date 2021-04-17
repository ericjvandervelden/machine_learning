import numpy as np
from IPython.display import display

from sklearn import datasets
iris=datasets.load_iris()

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn
seaborn.set()

if len(sys.argv)!=3:
    print("Use: log_reg_1_class_2_features_iris_geron_202102.py <features> <class>\n",file=sys.stderr) 
    sys.exit(1) 


#X=iris['data'][:,[2,3]]
features=np.array(sys.argv[1].split(',')).astype(int)
X=iris.data[:,features]

y = iris["target"]
#lg=LogisticRegression(C=10**10)
lg=LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10) # C=10 !
#lg=LogisticRegression(C=10)

lg.fit(X,y)

x0, x1 = np.meshgrid(
        np.linspace(0,8, 500).reshape(-1, 1),
        np.linspace(0,3.5, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = lg.predict_proba(X_new)
y_predict = lg.predict(X_new)


prb0=y_proba[:,0].reshape(x0.shape)
prb1=y_proba[:,1].reshape(x0.shape)
prb2=y_proba[:,2].reshape(x0.shape)
est=y_predict.reshape(x0.shape)


# print training data ,
plt.figure()
plt.plot(X[y==2,0],X[y==2,1],"g^",label="Iris-Virginia")
plt.plot(X[y==1,0],X[y==1,1],"bs",label="Iris-Versicolor")
plt.plot(X[y==0,0],X[y==0,1],"yo",label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap=ListedColormap(['#fafab0','#9898ff','#a0faa0'])
plt.contourf(x0,x1,est,cmap=custom_cmap)

countour1=plt.contour(x0,x1,prb1,cmap=plt.cm.brg)
countour0=plt.contour(x0,x1,prb0,cmap=plt.cm.brg)
countour2=plt.contour(x0,x1,prb2,cmap=plt.cm.brg)

plt.clabel(countour1,inline=1,fontsize=12)
plt.clabel(countour0,inline=1,fontsize=12)
plt.clabel(countour0,inline=1,fontsize=12)

plt.show()

