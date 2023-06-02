"""Initialize Session"""
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



"""Load data"""
file_path = 'oasis_cross-sectional_without_extra.csv'
data = pd.read_csv(file_path)

"""Filter Empty Data"""

data=data.dropna()
print(data)

y = data.CDR
features = ['M/F', 'Educ', 'Age', 'MMSE', 'nWBV']
X = data[features]
print(X.head())

features_mf = ['M/F']
X_mf = data[features_mf]
print(X_mf.head())

features_educ = ['Educ']
X_educ = data[features_educ]
print(X_educ.head())

features_age = ['Age']
X_age = data[features_age]
print(X_age.head())

features_mmse = ['MMSE']
X_mmse = data[features_mmse]
print(X_mmse.head())

features_volume = ['nWBV']
X_volume = data[features_volume]
print(X_volume.head())

features_not_educ = ['M/F', 'Age', 'MMSE', 'nWBV']
X_not_educ = data[features_not_educ]
print(X_not_educ.head())

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_full = RandomForestRegressor(random_state=1)
model_full.fit(train_X, train_y)

"""Trained model above"""

"""Full model Prediction"""
preds = model_full.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""Male/Female Prediction"""

train_X, val_X, train_y, val_y = train_test_split(X_mf, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_fm = RandomForestRegressor(random_state=1)
model_fm.fit(train_X, train_y)
preds = model_fm.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""Education Prediction"""

train_X, val_X, train_y, val_y = train_test_split(X_educ, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_educ = RandomForestRegressor(random_state=1)
model_educ.fit(train_X, train_y)
preds = model_educ.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""Age Prediction"""

train_X, val_X, train_y, val_y = train_test_split(X_age, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_age = RandomForestRegressor(random_state=1)
model_age.fit(train_X, train_y)
preds = model_age.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""MMSE prediction"""

train_X, val_X, train_y, val_y = train_test_split(X_mmse, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_mmse = RandomForestRegressor(random_state=1)
model_mmse.fit(train_X, train_y)
preds = model_mmse.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""Volume Prediciton"""

train_X, val_X, train_y, val_y = train_test_split(X_volume, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_volume = RandomForestRegressor(random_state=1)
model_volume.fit(train_X, train_y)
preds = model_volume.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))

"""Full Except for Education"""

train_X, val_X, train_y, val_y = train_test_split(X_not_educ, y,random_state = 0)
print("Target:")
print(train_X.head())
print("Feature: ")
print(train_y.head())

model_not_educ = RandomForestRegressor(random_state=1)
model_not_educ.fit(train_X, train_y)
preds = model_not_educ.predict(val_X)
print(val_y, preds)
print(mean_absolute_error(val_y, preds))
