SEED = 2137

import pandas as pd
import numpy as np

data = pd.read_csv("data/regresja/domy.csv", decimal='.')
data[data == '?'] = pd.NA
print(data.info())

data = data.drop("Id", axis="columns")
data = data[data["GrLivArea"] < 4_000]

def fill_missing(column, method='interp'):
    if method == 'fill':
        data[column] = data[column].ffill()
        data[column] = data[column].bfill()
    else:
        data[column] = data[column].interpolate()

def qual_to_val(column):
    qual_to_val = {"Ex": "6", "Gd": "5", "TA": "4", "Fa": "3", "Po": "2", "NA": "0", pd.NA: "0"}
    data[column] = data[column].map(qual_to_val)
    data[column] = pd.to_numeric(data[column])
    fill_missing(column)

for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]:
    qual_to_val(col)

def pytajniki(column):
    data[column] = pd.to_numeric(data[column])
    fill_missing(column)

for col in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
    pytajniki(col)

data["MSSubClass"] = data["MSSubClass"].astype(str)

data = pd.get_dummies(data, dtype=int)

y = data["SalePrice"]
X = data.drop("SalePrice", axis="columns")


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score

reg = linear_model.LinearRegression()
_ = reg.fit(X, y)

y_pred = reg.predict(X)
mean_squared_error(y, y_pred)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score

reg = linear_model.Ridge(alpha=.5)
_ = reg.fit(X, y)

y_pred = reg.predict(X)
mean_squared_error(y, y_pred)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
reg = linear_model.Ridge(alpha=.5)
_ = reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))
y_pred_ridge = reg.predict(X_test)

reg = linear_model.LinearRegression()
_ = reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

y_pred_linear = reg.predict(X_test)

from random import random, randint
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV

reg = linear_model.Ridge(alpha=1.0)
result = cross_validate(reg, X, y)
print(result["test_score"])
print(sum(result["test_score"]) / 5)

from sklearn.model_selection import train_test_split, cross_validate

reg = linear_model.LinearRegression()
result = cross_validate(reg, X, y)
print(result["test_score"])
print(sum(result["test_score"]) / 5)


import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_linear, color="red")
plt.scatter(y_test, y_pred_ridge, color="green")

plt.plot([0, 500_000], [0, 500_000], color="blue")
plt.show()
