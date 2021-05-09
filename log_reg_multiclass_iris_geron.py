from sklearn import datasets
iris=datasets.load_iris()

from sklearn.linear_model import LogisticRegression

X=iris['data'][:,[3]]
t=(iris['target']==2).astype(int)
log_reg = LogisticRegression(multi_class="multinomial",solver="newton-cg", C=10**10)
log_reg.fit(X,t)
display(log_reg.coef_)
display(log_reg.intercept_)
