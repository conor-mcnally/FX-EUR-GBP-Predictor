import csv
import numpy as np
import pandas as pd
import sklearn
import time
from datetime import date, timedelta
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#Load the data
file = open('EUR-GBP.csv', 'r')

#Store data in dataframe
df = pd.read_csv('EUR-GBP.csv')

#Some console information
print('\nLast 3 entries\n')
print(df.tail(3))
print("\nNumber of 15 min segments: ", (df.shape[0]))

#Plot the data
plt.figure(figsize=(16,8))
plt.title('EUR/GBP')
plt.xlabel('15 Min Segments')
plt.ylabel('Close Price')
plt.plot(df['4. close'])
plt.show()

#Get the close price
df = df[['4. close']]

#Display some console info
print('\nFirst 3 close prices\n')
print(df.head(4))

#Create a variable to predict 'x' days out into the future
future_days = 25

#Create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['4. close']].shift(-future_days)
print()
print(df.tail(4))

#Create the feature data set (X) and convert it to a numpy array and remove the last 'x' rows/days
feature_dataset = np.array(df.drop(['Prediction'], 1))[:-future_days]
print('\nFeature Dataset (First and Last 3)\n')
print(feature_dataset[:3])
print(feature_dataset[-3:])

#Create the target data set (y) and covert it ot a numpy array and get all of the target values except the last 'x' rows/days
target_dataset = np.array(df['Prediction'])[:-future_days]
print('\nTarget Dataset (First and Last 3)\n')
print(target_dataset[:3])
print(target_dataset[-3:])

#Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(feature_dataset, target_dataset, test_size = 0.25)

#Create the models
#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)

#get the last 'x' rows of the feature dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future

#Show the model tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#Show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

#Visualise the data
predictions = tree_prediction

valid = df[feature_dataset.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD $')
plt.plot(df['4. close'])
plt.plot(valid[['4. close', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
plt.show()

#Visualise the data
predictions = lr_prediction

valid = df[feature_dataset.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD $')
plt.plot(df['4. close'])
plt.plot(valid[['4. close', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
plt.show()
