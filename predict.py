import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import csv into dataframe
df = pd.read_csv("housing.csv")

# handle missing values
imputer = SimpleImputer(strategy="mean")
imputer = ColumnTransformer([("imputer", imputer, ["total_bedrooms"])])
df['total_bedrooms'] = imputer.fit_transform(df)

# handle one hot encoding values
ohe = pd.get_dummies(df['ocean_proximity'])
df = pd.concat([df, ohe], axis=1)
df.drop("ocean_proximity", axis=1, inplace=True)

# split data for training and testing
X = df.drop("median_house_value", axis=1)
y = df['median_house_value']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

RandForest = RandomForestRegressor()
score = cross_val_score(RandForest, X_train, y_train,
                        scoring="neg_mean_squared_error", cv=5)

# parameter tuning
params = {'n_estimators': [3,10,20,50], 'max_features': [2,3,4,10]}
grid_search = GridSearchCV(RandForest, params, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
predictions = grid_search.best_estimator_.predict(X_test)
comparison = pd.DataFrame(
    {"Test": y_test[:10].values, "Predictions": predictions[:10]})

# evaluate 
def regression_evaluation(predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    print(
        f"Mean Absolute Error: {mae} \nMean Squared Error: {mse} \nRoot Mean Squared Error: {rmse} \nR2 score: {r_squared}")
regression_evaluation(predictions)

# save model
filename = 'final_model.sav'
pickle.dump(grid_search, open(filename, "wb"))
