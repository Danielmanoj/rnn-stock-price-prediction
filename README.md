# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

![318182423-b1b15619-b3f9-4062-9ae9-cbb62c72497a](https://github.com/Danielmanoj/rnn-stock-price-prediction/assets/69635071/e17b741d-cf6c-4a0a-815f-30940a0eeaf1)

## Design Steps

### Step 1:
import required header files


### Step 2:
read the csv file using pd.read_csv

### Step 3:
use minmaxscaler to set range of feature

### Step 4:
train the dataset

### Step 5:
compile the training set

### Step 6:
fit the training set



## Program
#### Name: MANOJ G
#### Register Number: 212222240060

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()
model.add(layers.SimpleRNN(60,input_shape=(60,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam',loss='mse')


model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
print('MANOJ G')
print('REG.NO:212222240060')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time
![image](https://github.com/Danielmanoj/rnn-stock-price-prediction/assets/69635071/6e657685-9e0f-421c-a467-de0ba35f6b0e)


### Mean Square Error
![318182705-5e5c54e1-b0f4-42a5-8d5f-d26899daf8f4](https://github.com/Danielmanoj/rnn-stock-price-prediction/assets/69635071/6ac10ca9-6964-4f0f-93cf-9253c7a554f7)


## Result
Thus the program is successfully created and executed.
