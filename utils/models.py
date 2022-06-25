import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from pandas import DataFrame, date_range, concat
from neuralprophet import NeuralProphet
from math import pi, cos
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

Y_train=pd.read_csv("./dat/train.csv")
X_train=pd.read_csv("./dat/xtrain.csv")

# X_train['Date']=pd.to_datetime(X_train['Date'])
# X_train['Hour'] = X_train['Date'].dt.hour
# X_train['Day'] = X_train['Date'].dt.day
# X_train['Month'] = X_train['Date'].dt.month
# X_train['Price-1']=Y_train['Price'].shift(1)
# X_train.drop(columns=['Date'],inplace=True)
# X_train.dropna(0)
# X_train.to_csv('xtrain.csv')

# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(X_train, Y_train)
# shap_values = shap.TreeExplainer(clf).shap_values(X_train)
# plt.figure(111)
# shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
# plt.savefig('Features.png')
process = RobustScaler()
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)

def neural(a,b,c,d,e,f):
    
    model1= MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10))
    model1.fit(x_train,y_train)
    value=model1.predict([[a,b,c,d,e,f],] )  
    return value
def rf(a,b,c,d,e,f):
    model2 = RandomForestRegressor(bootstrap=True, min_samples_leaf=1,
                               n_estimators=20, min_samples_split=15, max_features='sqrt', max_depth=20)
    model2.fit(x_train,y_train)
    value=model2.predict([[a,b,c,d,e,f],] )  
    return value
def gb(a,b,c,d,e,f):
    model3 = GradientBoostingRegressor()
    model3.fit(x_train,y_train)
    value=model3.predict([[a,b,c,d,e,f],] )  
    return value