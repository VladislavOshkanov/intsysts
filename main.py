import csv
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np 

data = []

df_train = pd.read_csv('/home/voshkanov/house-prices-datasets/train.csv', index_col='Id')
df_test = pd.read_csv('/home/voshkanov/house-prices-datasets/test.csv', index_col='Id')

# print (df_train)


print('Categorical: ', df_train.select_dtypes(include=['object']).columns)

#numerical features (see comment about 'MSSubCLass' here above)
print('Numerical: ', df_train.select_dtypes(exclude=['object']).columns)