#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


df = pd.read_csv('..\DATA\Frozen_Dessert_Production.csv', parse_dates=True, index_col = 'DATE')


# In[112]:


df.head()


# In[113]:


df.info()


# In[114]:


df.plot(figsize = (12,8))


# In[115]:


len(df)


# In[116]:


test_size = 24


# In[117]:


test_ind = len(df)-test_size


# In[118]:


train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


# In[119]:


len(train)


# In[120]:


len(test)


# In[121]:


from sklearn.preprocessing import MinMaxScaler


# In[122]:


scaler = MinMaxScaler()


# In[123]:


scaler.fit(train)


# In[124]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[125]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[126]:


length = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
val_generator = TimeseriesGenerator(scaled_test, scaled_test, length = length, batch_size=1)


# In[127]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# In[128]:


n_features=1


# In[129]:


model = Sequential()

model.add(LSTM(150, activation='relu', input_shape = (length, n_features)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[130]:


from tensorflow.keras.callbacks import EarlyStopping


# In[131]:


early = EarlyStopping(monitor = 'val_loss', patience=3)


# In[132]:


model.fit(generator,
          epochs=50,
         validation_data = val_generator,
          callbacks = [early]
         )


# In[133]:


model.summary()


# In[134]:


loss = pd.DataFrame(model.history.history)


# In[135]:


loss.plot()


# In[136]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[137]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[138]:


test['LSTM predictions'] = true_predictions


# In[139]:


test.head()


# In[140]:


test.plot()


# In[ ]:





# In[141]:


scaler = MinMaxScaler()


# In[144]:


df_scale = scaler.fit_transform(df)


# In[145]:


length=12


# In[146]:


full_gen = TimeseriesGenerator(df_scale, df_scale, length = length, batch_size=1)


# In[148]:


model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(full_gen,epochs=5)


# In[149]:


forecast = []
period=12
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(period):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    forecast.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[150]:


forecast = scaler.inverse_transform(forecast)


# In[152]:


forecast


# In[153]:


test


# In[155]:


forecast_ind = pd.date_range(start='2019-10-01',periods=period,freq='MS')


# In[157]:


forecast_df = pd.DataFrame(data=forecast,index=forecast_ind,
                           columns=['Forecast'])


# In[158]:


forecast_df


# In[160]:


df.plot()
forecast_df.plot()


# In[161]:


ax = df.plot(figsize=(14,8))
forecast_df.plot(ax=ax)


# In[ ]:




