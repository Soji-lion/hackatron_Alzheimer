import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

file_path = 'oasis_cross-sectional.csv'
data = pd.read_csv(file_path)

data=data.dropna()
print(data)

y = data.CDR
features = ['M/F', 'Age', 'Educ', 'MMSE', 'nWBV']
X = data[features]
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(val_X)
print(val_y, preds)
