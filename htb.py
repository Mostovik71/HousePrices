import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
train = pd.read_csv('data/train.csv', index_col='Id')
test = pd.read_csv('data/test.csv', index_col='Id')
print(train.shape) #80 columns
train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)# A lot of NaN values
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
y = train['SalePrice']
train.drop('SalePrice', inplace=True, axis=1)
train.drop(['MSZoning', 'Street', 'LandContour', 'PavedDrive', '3SsnPorch',
            'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'BsmtCond',
            'BsmtFinSF2', 'Heating', 'CentralAir', 'Electrical', 'LowQualFinSF',
            'KitchenAbvGr', 'Functional', 'GarageQual', 'GarageCond', 'ScreenPorch',
            'PoolArea', 'MiscVal'], inplace=True, axis=1) #A lot of same values
train.drop(['LotShape', 'LotConfig', 'EnclosedPorch', 'SaleType', 'SaleCondition',
            'Condition1', 'BldgType', 'RoofStyle', 'ExterCond', 'BsmtFinType2'], inplace=True, axis=1) #A lot of values, but probably benefit
train.drop(['WoodDeckSF', 'OpenPorchSF'], inplace=True, axis=1) #MAYBE, MAYBE
print(tabulate(train.head(), headers='keys'))
print(train.shape) #40 columns


# x = train
# X_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, train_size=0.8)
# cat_only = X_train.select_dtypes(include=['object'])
