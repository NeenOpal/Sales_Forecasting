   
import numpy as np
import io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
dataset_train = pd.read_csv('V:\\Accuracy_Neen\\Temp\\P.csv')
training_set = dataset_train.iloc[:,0:26].values
#training_set=training_set.astype(np.float)

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(training_set[:,0:])
training_set[:,0:]=imputer.transform(training_set[:,0:])


training_predictor=dataset_train.iloc[:,26:27].values

imputer1=imputer.fit(training_predictor[:,0:])
training_predictor[:,0:]=imputer.transform(training_predictor[:,0:])

training_set[training_set<0]=0
training_predictor[training_predictor<0]=0

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled_predictor=sc.fit_transform(training_predictor)

X_train =[]
y_train = []
for i in range(218011):
    X_train.append(training_set_scaled[i,0:26])
    y_train.append(training_set_scaled_predictor[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 512, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.5))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 60, batch_size = 144)


real_test_sales=training_predictor[218011:,0]
inputs=training_set_scaled[218011:,0:26]

X_test = []
for i in range(55939):
    X_test.append(inputs[i,0:26])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

predicted_test_sales = regressor.predict(X_test)
predicted_test_sales = sc.inverse_transform(predicted_test_sales)

from sklearn.metrics import r2_score
print(r2_score(real_test_sales,predicted_test_sales))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(real_test_sales,predicted_test_sales))