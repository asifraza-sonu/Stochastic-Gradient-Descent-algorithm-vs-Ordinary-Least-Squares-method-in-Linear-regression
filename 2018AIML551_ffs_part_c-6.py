import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset = pd.read_csv('concrete_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1:].values
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
D = []
reg = LinearRegression()
for i in range (0,8) :
    K1 = X_train.iloc[:,i].values.reshape(-1,1)
    reg.fit(K1,y_train.ravel())
    print('Model :',i,' the Co-efficient obtained is :',reg.coef_)
    print('Model',i,'the Intercept obtained is :', reg.intercept_)
    
    K2 = X_test.iloc[:,i].values.reshape(-1,1)

    y_predict = reg.predict(K2)
    MSE = metrics.mean_squared_error(y_test,y_predict)
    RMSE = np.sqrt(MSE)
    N = len(y_test)
    RSE = np.sqrt((N* MSE) /(N-2))
    R2_Score = reg.score(K2,y_test.ravel())

    print('R2_score:',R2_Score)
    print('Residual Squared error for Model',i,':',RSE)

D.insert(i,R2_Score)
print('\n')

i+=1
print(max(D))
index = D.index(max(D))

print('R2 value is higher and optimal , when the number of features are:',index+1)