#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Input
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, LSTM, GRU
from keras.layers import Conv1D
from keras.layers import MaxPooling1D 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# In[2]:


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# In[3]:


df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')


# In[5]:


df


# In[6]:


# take three day of datas to predict one day
timestep = 72
# size of training data - 500 days (12000 hours)
training_num = 12000

epoch = 10
batch_size = 200


# In[7]:


dt = df['T (degC)']
dt = np.array(dt).reshape(-1,1)
dt.shape


# # <font color = purple>Normalize the data to (0, 1)</font>

# In[8]:


sc = MinMaxScaler(feature_range = (0, 1))
dt_nor = sc.fit_transform(dt)
dt_nor


# # <font color = purple>Find the best width</font>

# In[9]:


# take three day of datas to predict one day
timestep
# size of training data - 500 days
training_num = 12000

epoch = 10
batch_size = 200

def width(timestep,model_kind):
  xTrainSet = dt_nor[:training_num]
  yTrainSet = dt_nor[1:training_num+1]

  xTrain = []
  for i in range(timestep, training_num):
    xTrain.append(xTrainSet[i-timestep : i])
  xTrain = np.array(xTrain)
  xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

  yTrain = []
  for i in range(timestep, training_num):
    yTrain.append(yTrainSet[i])
  yTrain = np.array(yTrain)

  if model_kind == 'model_rnn':
    model = Sequential()
    model.add(LSTM(128, return_sequences = True, input_shape = (xTrain.shape[1],1)))
    model.add(GRU(64))
    model.add(Dense(1))

  if model_kind == 'model_dense':
    model = Sequential()
    model.add(Input(shape = (xTrain.shape[1])))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

  if model_kind == 'model_cnn':
    conv_width = 3
    model = Sequential()
    model.add(Conv1D(64, kernel_size=(conv_width), input_shape = (xTrain.shape[1],1), activation='relu'))
    # model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(32, kernel_size=(conv_width), activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
  
  model.compile(optimizer = 'adam',
         loss = 'mean_squared_error',
         metrics = [tf.metrics.MeanAbsoluteError()])
  model.fit(x = xTrain, y = yTrain, epochs = epoch, batch_size = batch_size, verbose=0)

  xTestSet = dt_nor[training_num : 40800-2]
  xTestSet = np.array(xTestSet)

  yTestSet = dt_nor[training_num+1 : 40800-1]
  yTestSet = np.array(yTestSet)

  xTest = []
  for i in range(timestep, len(xTestSet)):
      xTest.append(xTestSet[i-timestep : i])
  xTest = np.array(xTest)

  yTest = []
  for i in range(timestep, len(xTestSet)):
      yTest.append(yTestSet[i])
  yTest = np.array(yTest)
  yTest = sc.inverse_transform(yTest)

  yPredictes = model.predict(x=xTest)
  yPredictes = sc.inverse_transform(yPredictes)

  r2 = r2_score(yTest, yPredictes)
  return r2


# In[10]:


rnn_width_dict = {}
for step in range(5,51):
  rnn_width_dict[step] = width(step,'model_rnn')
  print(step,end="-")
  if step%10 == 0:
    print()
rnn_width = max(rnn_width_dict,key=rnn_width_dict.get)
rnn_width


# In[11]:


dense_width_dict = {}
for step in range(5,51):
  dense_width_dict[step] = width(step,'model_dense')
  print(step,end="-")
  if step%10 == 0:
    print()
dense_width = max(dense_width_dict,key=dense_width_dict.get)
dense_width


# In[12]:


cnn_width_dict = {}
for step in range(5,51):
  cnn_width_dict[step] = width(step,'model_cnn')
  print(step,end="-")
  if step%10 == 0:
    print()
cnn_width = max(cnn_width_dict,key=cnn_width_dict.get)
cnn_width


# # <font color = purple>Rnn</font>

# In[13]:


timestep = rnn_width


# In[14]:


xTrainSet = dt_nor[:training_num]
yTrainSet = dt_nor[1:training_num+1]


# In[15]:


xTrain = []
for i in range(timestep, training_num):
  xTrain.append(xTrainSet[i-timestep : i])
xTrain = np.array(xTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
print(xTrain.shape)

yTrain = []
for i in range(timestep, training_num):
  yTrain.append(yTrainSet[i])
yTrain = np.array(yTrain)
print(yTrain.shape)


# In[16]:


model_rnn = Sequential()
model_rnn.add(LSTM(128, return_sequences = True, input_shape = (xTrain.shape[1],1)))
model_rnn.add(GRU(64))
model_rnn.add(Dense(1))

model_rnn.summary()


# In[17]:


model_rnn.compile(optimizer = 'adam',
         loss = 'mean_squared_error',
         metrics = [tf.metrics.MeanAbsoluteError()])
model_rnn.fit(x = xTrain, y = yTrain, epochs = epoch, batch_size = batch_size)


# # <font color = purple>Test model's accuracy by r2_score (1200 days)</font>

# In[18]:


xTestSet = dt_nor[training_num : 40800-2]
xTestSet = np.array(xTestSet)

yTestSet = dt_nor[training_num+1 : 40800-1]
yTestSet = np.array(yTestSet)


# In[19]:


xTest = []
for i in range(timestep, len(xTestSet)):
    xTest.append(xTestSet[i-timestep : i])
xTest = np.array(xTest)
print(len(xTest))

yTest = []
for i in range(timestep, len(xTestSet)):
    yTest.append(yTestSet[i])
yTest = np.array(yTest)
yTest = sc.inverse_transform(yTest)
len(yTest)


# In[20]:


yPredictes = model_rnn.predict(x=xTest)
yPredictes = sc.inverse_transform(yPredictes)
yPredictes


# In[21]:


r2_value = {}


# In[22]:


plt.plot(yTest, 'c-', label='Real')
plt.plot(yPredictes, 'm-', label='Predict')
# plt.plot(data_original, color='red', label='Real')
# plt.plot(range(len(y_train)),yPredicts, color='blue', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2 = r2_score(yTest, yPredictes)
r2_value['RNN'] = r2
print(r2)


# # <font color=purple>Dense</font>

# In[23]:


timestep = dense_width


# In[24]:


xTrainSet = dt_nor[:training_num]
yTrainSet = dt_nor[1:training_num+1]


# In[25]:


xTrain = []
for i in range(timestep, training_num):
  xTrain.append(xTrainSet[i-timestep : i])
xTrain = np.array(xTrain)
#xTrain = np.squeeze(xTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
print(xTrain.shape)

yTrain = []
for i in range(timestep, training_num):
  yTrain.append(yTrainSet[i])
yTrain = np.array(yTrain)
#yTrain = np.reshape(yTrain, (yTrain.shape[0], 1))
print(yTrain.shape)


# In[26]:


model_dense = Sequential()
model_dense.add(Input(shape = (xTrain.shape[1])))
model_dense.add(Flatten())
model_dense.add(Dense(128, activation='relu'))
model_dense.add(Dense(64, activation='relu'))
model_dense.add(Dense(1))

model_dense.summary()


# In[27]:


model_dense.compile(optimizer = 'adam',
          loss = 'mean_squared_error',
          metrics = [tf.metrics.MeanAbsoluteError()])
model_dense.fit(x = xTrain, y = yTrain, epochs = epoch, batch_size = batch_size)


# # <font color=purple>Test model's accuracy by r2_score (1200 days)</font>

# In[28]:


xTestSet = dt_nor[training_num : 40800-2]
xTestSet = np.array(xTestSet)

yTestSet = dt_nor[training_num+1 : 40800-1]
yTestSet = np.array(yTestSet)


# In[29]:


xTest = []
for i in range(timestep, len(xTestSet)):
    xTest.append(xTestSet[i-timestep : i])
xTest = np.array(xTest)
#xTest = np.squeeze(xTest)

yTest = []
for i in range(timestep, len(xTestSet)):
    yTest.append(yTestSet[i])
yTest = np.array(yTest)
yTest = sc.inverse_transform(yTest)
len(xTest)


# In[30]:


yTest.shape


# In[31]:


yPredictes = model_dense.predict(x=xTest)
yPredictes = sc.inverse_transform(yPredictes)
yPredictes


# In[32]:


plt.plot(yTest, 'c-', label='Real')
plt.plot(yPredictes, 'm-', label='Predict')
# plt.plot(data_original, color='red', label='Real')
# plt.plot(range(len(y_train)),yPredicts, color='blue', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2 = r2_score(yTest, yPredictes)
r2_value['Dense'] = r2
print(r2)


# # <font color=purple>Cnn</font>

# In[33]:


timestep = cnn_width


# In[34]:


xTrainSet = dt_nor[:training_num]
yTrainSet = dt_nor[1:training_num+1]


# In[35]:


xTrain = []
for i in range(timestep, training_num):
  xTrain.append(xTrainSet[i-timestep : i])
xTrain = np.array(xTrain)
# xTrain = np.squeeze(xTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
print(xTrain.shape)

yTrain = []
for i in range(timestep, training_num):
  yTrain.append(yTrainSet[i])
yTrain = np.array(yTrain)
#yTrain = np.reshape(yTrain, (yTrain.shape[0], 1))
print(yTrain.shape)


# In[36]:


conv_width = 3
model_cnn = Sequential()
model_cnn.add(Conv1D(64, kernel_size=(conv_width),  input_shape = (xTrain.shape[1],1), activation='relu'))
model_cnn.add(Conv1D(32, kernel_size=(conv_width), activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(32, activation='relu'))
model_cnn.add(Dense(1))

model_cnn.summary()


# In[37]:


model_cnn.compile(optimizer = 'adam',
          loss = 'mean_squared_error',
          metrics = [tf.metrics.MeanAbsoluteError()])
model_cnn.fit(x = xTrain, y = yTrain, epochs = epoch, batch_size = batch_size)


# # <font color=purple>Test model's accuracy by r2_score (1200 days)</font>

# In[38]:


xTestSet = dt_nor[training_num : 40800-2]
xTestSet = np.array(xTestSet)

yTestSet = dt_nor[training_num+1 : 40800-1]
yTestSet = np.array(yTestSet)


# In[39]:


xTest = []
for i in range(timestep, len(xTestSet)):
    xTest.append(xTestSet[i-timestep : i])
xTest = np.array(xTest)
# xTest = np.squeeze(xTest)

yTest = []
for i in range(timestep, len(xTestSet)):
    yTest.append(yTestSet[i])
yTest = np.array(yTest)
yTest = sc.inverse_transform(yTest)
len(xTest)


# In[40]:


yPredictes = model_cnn.predict(x=xTest)
yPredictes = sc.inverse_transform(yPredictes)
yPredictes.shape


# In[41]:


plt.plot(yTest, 'c-', label='Real')
plt.plot(yPredictes, 'm-', label='Predict')
# plt.plot(data_original, color='red', label='Real')
# plt.plot(range(len(y_train)),yPredicts, color='blue', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2 = r2_score(yTest, yPredictes)
r2_value['CNN'] = r2
print(r2)


# # <font color=purple>Compare</font>

# In[42]:


r2_value


# In[43]:


x = np.arange(3)
width = 0.2

val_r2 = r2_value.values()

plt.figure(figsize=(4,7))
plt.ylabel('r2_score [T (degC)]')
plt.bar(x , val_r2, 0.4, label='Test')
plt.xticks(ticks=x, labels=r2_value.keys(), rotation=45)
_ = plt.legend()


# # <font color=purple>Use predict data to predict future</font>

# # <font color=purple>Rnn</fnt>

# In[44]:


# Take last 24 hours to predict
Predict_hours = 24

xPred = dt_nor[-Predict_hours-rnn_width:-Predict_hours]
xPred = np.array(xPred)
xPred_in = sc.inverse_transform(xPred)
xPred = np.reshape(xPred, (xPred.shape[1], xPred.shape[0], 1))

yFutureTest = dt_nor[-Predict_hours:]
yFutureTest = np.array(yFutureTest)
yFutureTest = sc.inverse_transform(yFutureTest)

real = []
real = np.append(xPred_in, yFutureTest, axis = 0)


# In[45]:


xPred.shape


# In[46]:


def PredFuture_rnn(xPred):
  yPred = model_rnn.predict(x=xPred)
  yPred = np.reshape(yPred, (1, 1, 1))
  data = np.append(xPred, yPred, axis = 1)
  data = data[:, -(rnn_width):, :]
  return data

yModelPred = []
for i in range (Predict_hours):
  xPred = PredFuture_rnn(xPred)
  yModelPred.append(xPred[0][-1])


# In[47]:


yModelPred = np.array(yModelPred)
yModelPred = sc.inverse_transform(yModelPred)


# In[48]:


plt.plot(real, 'y-', label='train')
plt.plot(range(rnn_width, Predict_hours + rnn_width), yFutureTest, 'c-', label='Real')
plt.plot(range(rnn_width, Predict_hours + rnn_width), yModelPred, 'm-', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2_future = {}
r2 = r2_score(yFutureTest, yModelPred)
r2_future['RNN'] = r2
print(r2)


# # <font color=purple>Cnn</font>

# In[49]:


# Take last 24 hours to predict
Predict_hours = 24

xPred = dt_nor[-Predict_hours-cnn_width:-Predict_hours]
xPred = np.array(xPred)
xPred_in = sc.inverse_transform(xPred)
# #xPred = np.reshape(xPred, (1,xPred.shape[0],xPred.shape[1]))
xPred = np.reshape(xPred, (xPred.shape[1], xPred.shape[0], 1))

yFutureTest = dt_nor[-Predict_hours:]
yFutureTest = np.array(yFutureTest)
yFutureTest = sc.inverse_transform(yFutureTest)

real = []
real = np.append(xPred_in, yFutureTest, axis = 0)


# In[50]:


def PredFuture_cnn(xPred):
  yPred = model_cnn.predict(xPred)
  yPred = np.reshape(yPred, (1, 1, 1))
  data = np.append(xPred, yPred, axis = 1)
  data = data[:, -(cnn_width):, :]
  return data

yModelPred = []
for i in range (Predict_hours):
  xPred = PredFuture_cnn(xPred)
  yModelPred.append(xPred[0][-1])


# In[51]:


yModelPred = np.array(yModelPred)
yModelPred = sc.inverse_transform(yModelPred)


# In[52]:


plt.plot(real, 'y-', label='train')
plt.plot(range(cnn_width, Predict_hours + cnn_width), yFutureTest, 'c-', label='Real')
plt.plot(range(cnn_width, Predict_hours + cnn_width), yModelPred, 'm-', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2 = r2_score(yFutureTest, yModelPred)
r2_future['CNN'] = r2
print(r2)


# # <font color=purple>Dense</font>

# In[53]:


# Take last 24 hours to predict
Predict_hours = 24

xPred = dt_nor[-Predict_hours-dense_width:-Predict_hours]
xPred = np.array(xPred)
xPred_in = sc.inverse_transform(xPred)
# #xPred = np.reshape(xPred, (1,xPred.shape[0],xPred.shape[1]))
xPred = np.reshape(xPred, (xPred.shape[1], xPred.shape[0], 1))

yFutureTest = dt_nor[-Predict_hours:]
yFutureTest = np.array(yFutureTest)
yFutureTest = sc.inverse_transform(yFutureTest)

real = []
real = np.append(xPred_in, yFutureTest, axis = 0)


# In[54]:


def PredFuture_dense(xPred):
  yPred = model_dense.predict(xPred)
  yPred = np.reshape(yPred, (1, 1, 1))
  data = np.append(xPred, yPred, axis = 1)
  data = data[:, -(dense_width):, :]
  return data

yModelPred = []
for i in range (Predict_hours):
  xPred = PredFuture_dense(xPred)
  yModelPred.append(xPred[0][-1])


# In[55]:


yModelPred = np.array(yModelPred)
yModelPred = sc.inverse_transform(yModelPred)


# In[56]:


plt.plot(real, 'y-', label='train')
plt.plot(range(dense_width, Predict_hours + dense_width), yFutureTest, 'c-', label='Real')
plt.plot(range(dense_width, Predict_hours + dense_width), yModelPred, 'm-', label='Predict')
plt.title(label='Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='T')
plt.legend()
plt.show()

r2 = r2_score(yFutureTest, yModelPred)
r2_future['Dense'] = r2
print(r2)


# # <font color=purple>Compare</font>

# In[57]:


r2_future


# In[58]:


x = np.arange(3)
width = 0.2

val_r2 = r2_future.values()

plt.figure(figsize=(4,7))
plt.ylabel('r2_score [T (degC)]')
plt.axhline(0, color= 'r')
plt.bar(x , val_r2, 0.4, label='Test')
plt.xticks(ticks=x, labels=r2_future.keys(), rotation=45)
_ = plt.legend()

