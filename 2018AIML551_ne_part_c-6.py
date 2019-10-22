import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
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
Reg = LinearRegression()
Reg.fit(X_train,y_train.ravel())
y_predict = Reg.predict(X_test)
#To print the obtained weights of the model
dataset_index = list(dataset.columns)
Obtained_weights = list (Reg.coef_)
i= 0
for W in dataset_index[:-1] :
    print ('The Obtained weight of feature',W ,'is' ,Obtained_weights[i])
    i+=1
R2_Score = Reg.score(X_train,y_train)
MAE = metrics.mean_absolute_error(y_test,y_predict)
MSE = metrics.mean_squared_error(y_test,y_predict)
RMSE = np.sqrt(MSE)
N = len(y_test)
RSE = np.sqrt((N* MSE) /(N-2))
print('Predicted intercept value is: ', Reg.intercept_)
print('Predicted  coeffient value is: ',Reg.coef_ )
print('Mean absolute error (MAE) is :',MAE)
print('Mean squared error (MSE)) is :',MSE)
print('Root Mean squared error(RMSE) is :',RMSE)
print('RSE - Residual Squared Error is :',RSE)
print ('metrics_R2Score is :',metrics.r2_score(y_test,y_predict))
print('R2 score is :',R2_Score)