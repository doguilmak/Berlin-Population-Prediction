# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:17:18 2021

@author: doguilmak

https://www.w3schools.com/python/python_ml_linear_regression.asp

dataset: https://www.macrotrends.net/cities/204296/berlin/population

"""
# %%
# Importing Generally Used Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import time

# %%
# Data Preprocessing

# Uploading Datas
start = time.time()
df = pd.read_csv('Berlin-population-2020-12-30.csv')
print(df.info()) # Looking for the missing values
print(df.describe())
print(df.isnull().sum())

# Creating correlation matrix heat map
"""
Plot rectangular data as a color-encoded matrix.
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""
plt.figure(figsize = (12, 6))
sns.heatmap(df.corr(), annot = True)
sns.pairplot(df)
plt.show()

# DataFrame Slice
x = df.iloc[:, 0:1] # Dates
y = df.iloc[:, 1:2] # Populations

# NumPy Array Translate
X = x.values
Y = y.values

# %%
# Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Plotting Linear Regression
plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color='red')
plt.plot(x, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel('Population')
sns.set_style("whitegrid")
plt.show()

# %%
# Polynomial Regression

# 2nd Order Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y)
# Plotting 2nd Order Polynomial
plt.figure(figsize=(16, 8))
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'green')
plt.title('2nd Order Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

# 4th Order Polynomial
poly_reg4 = PolynomialFeatures(degree = 4)
x_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, y)
plt.figure(figsize=(16, 8))
# Plotting 4th Order Polynomial 
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg4.predict(poly_reg4.fit_transform(X)), color='black')
plt.title('4th Order Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

#%%
# Population Predictions in Berlin in 2040

print('\nPopulation Predictions in Berlin in 2035:')
print("Linear Regression:")
print(lin_reg.predict([[2035]]))

print("\nPolinomal Regression(degree=2):")
print(lin_reg2.predict(poly_reg2.fit_transform([[2035]])))

print("\nPolinomal Regression(degree=4):")
print(lin_reg4.predict(poly_reg4.fit_transform([[2035]])))

# %%
# R² Values of the Regressions

"""
What is R²?
The r2_score function computes the coefficient of determination, usually 
denoted as R².

It represents the proportion of variance (of y) that has been explained by 
the independent variables in the model. It provides an indication of goodness 
of fit and therefore a measure of how well unseen samples are likely to be 
predicted by the model, through the proportion of explained variance.

As such variance is dataset dependent, R² may not be meaningfully comparable 
across different datasets. Best possible score is 1.0 and it can be negative 
(because the model can be arbitrarily worse). A constant model that always 
predicts the expected value of y, disregarding the input features, would get a 
R² score of 0.0.

https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

"""
print('\n\nR² Values of the Regressions:\n')
print('Linear R² value')
print(r2_score(Y, lin_reg.predict(X)))

print('\nPolynomial R² value(degree=2)')
print(r2_score(Y, lin_reg2.predict(poly_reg2.fit_transform(X))))

print('\nPolynomial R² value(degree=4)')
print(r2_score(Y, lin_reg4.predict(poly_reg4.fit_transform(X))))

# %%
# Calculating and Plotting Difference Between Actual Population and Predicted 
#                                                                   Population

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Making DataFrame
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['date',
                                                             ' Population'])
for i in range(0, len(data)):
    new_data['date'][i] = data['date'][i]
    new_data[' Population'][i] = data[' Population'][i]

# Setting Index
new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)

# Creating Train and Test Sets
dataset = new_data.values

# Slicing for Train
train = dataset[0:61, :]
valid = dataset[60:72, :]

# Converting Dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create and Fit the LSTM Network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, 
               input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=32, batch_size=1, verbose=2)

# Predicting Values, Using Past From the Train Data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
population_Pred = model.predict(X_test)
population_Pred = scaler.inverse_transform(population_Pred)

rms=np.sqrt(np.mean(np.power((valid-population_Pred), 2)))
print(f"RMS(Difference between actual population and predicted population): {rms}")

# %%

# Plotting
plt.figure(figsize=(16, 8))
train = new_data[:61]
valid = new_data[60:72]
valid['Predictions'] = population_Pred

plt.axis([2007, 2023, 3400000, 3800000])
plt.title('Berlin Population Prediction', fontsize=18)
plt.xlabel('Years', fontsize=18)
plt.ylabel(' Population', fontsize = 18)
plt.plot(train[' Population'])
plt.plot(valid[[' Population', 'Predictions']])
plt.legend(["Train Dataset", "Berlin Actual Population", "Berlin Predicted Population"])
plt.show()

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
