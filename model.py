
import numpy as np
import pandas as pd

dataset_train = pd.read_csv('V:\\Accuracy_Neen\\New_System\\Sales_Data_Train_New.csv')
training_set = dataset_train.iloc[:,0:17].values
training_predictor=dataset_train.iloc[:,17:18].values
#training_set_inde=dataset_train.iloc[:,3:4].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled_predictor=sc.fit_transform(training_predictor)
#training_set_scaled_inde=sc.fit_transform(training_set_inde)
X_train =[]
y_train = []
for i in range(13586):
    X_train.append(training_set_scaled[i,0:17])
    y_train.append(training_set_scaled_predictor[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 15, batch_size = 1)

#dataset_test = pd.read_csv('V:\\Accuracy_Neen\\New_System\\Sales_Data_Test_2yrs.csv')
real_test_sales=training_predictor[13586:,0]
inputs=training_set_scaled[13586:,0:17]
'''dataset_total = pd.concat((dataset_train['VALUE'],dataset_test['VALUE']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs= inputs.reshape(-1,1)
inputs = sc.transform(inputs)
'''
X_test = []
for i in range(2281):
    X_test.append(inputs[i,0:17])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

predicted_test_sales = regressor.predict(X_test)
predicted_test_sales = sc.inverse_transform(predicted_test_sales)

from sklearn.metrics import r2_score
print(r2_score(real_test_sales,predicted_test_sales))

import csv
with open("V:\\Accuracy_Neen\\New_System\\Temp.csv","w") as f:
    writer=csv.writer(f)
    writer.writerow(predicted_test_sales)