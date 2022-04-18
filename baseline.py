import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

pd.options.display.max_rows = 999
train = pd.read_csv('data/train.csv', index_col='Id')
test = pd.read_csv('data/test.csv', index_col='Id')
train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
y = train['SalePrice']
train.drop('SalePrice', inplace=True, axis=1)
x = train
X_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, train_size=0.8)
# print(train.isnull().sum())


scalartrain = X_train.select_dtypes(include=['int64', 'float64'])
scalartest = x_val.select_dtypes(include=['int64', 'float64'])  # = scalartest = test... for submission
scalartrain = scalartrain.apply(lambda x: x.fillna(x.mode()[0]))
scalartest = scalartest.apply(lambda x: x.fillna(x.mode()[0]))
forest_model = RandomForestRegressor(n_estimators=50)

forest_model.fit(scalartrain, y_train)
pred = forest_model.predict(scalartest)
print('RMSE:', mean_squared_error(np.log(y_val), np.log(pred), squared=False))

# For submission
# sub = pd.Series(pred, index=test.index, name='SalePrice')
#
# sub.to_csv('sub(17.04.2022)(1).csv')
