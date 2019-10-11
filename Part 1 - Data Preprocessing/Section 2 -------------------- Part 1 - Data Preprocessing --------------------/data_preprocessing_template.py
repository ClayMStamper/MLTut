# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
lastColumn = 3
y = dataset.iloc[:, lastColumn].values

# Taking care of missing data

from sklearn.preprocessing import Imputer

#replace missing data of index:X by the mean of the column
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#prevent machine learning models from attributing order to encoded data

encoder_X = LabelEncoder()
X[:,0] = encoder_X.fit_transform(X[:,0]) #returns 1st column of the matrix "X" encoded
onehot = OneHotEncoder(categorical_features=[0]); #specidy which column we want ot "onehotencode"
X = onehot.fit_transform(X).toarray()

encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)


#split the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

