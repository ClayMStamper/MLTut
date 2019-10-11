# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

depVar = 1
data = "Salary_Data.csv"

# Importing the dataset
dataset = pd.read_csv(data)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, depVar].values

#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler();
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict test set results
y_pred = regressor.predict(X_test)

#visualize training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')

#visualize test set
plt.scatter(X_test, y_test, color='green')


plt.show()














