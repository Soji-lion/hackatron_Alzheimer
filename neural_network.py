import pandas as pd
    
# Load data
file_path = '../oasis_cross-sectional.csv'
data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
data = data.dropna(axis=0)
# Choose target and features
y = data.CDR
features = ['M/F', 'Age', 
                        'Educ', 'MMSE', 'nWBV']
X = data[features]


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 256)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(val_X)
print(val_y, preds)