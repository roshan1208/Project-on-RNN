#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train = pd.read_csv('Google_Stock_Price_Train.csv', parse_dates=True, index_col='Date')


# In[3]:


df_test = pd.read_csv('Google_Stock_Price_Test.csv', parse_dates=True, index_col='Date')


# In[4]:


df_train.info()


# In[6]:


df_train.head()


# In[7]:


df_train['Close'] = df_train['Close'].apply(lambda x: float(''.join(x.split(','))))


# In[8]:


df_train['Volume'] = df_train['Volume'].apply(lambda x: float(''.join(x.split(','))))


# In[9]:


df_train.info()


# In[10]:


df_train['Volume'].plot()


# In[11]:


df_train['Open'].plot()


# In[12]:


df_test.info()


# In[13]:


df_test['Volume'] = df_test['Volume'].apply(lambda x: float(''.join(x.split(','))))


# In[14]:


test_size = len(df_test)


# In[15]:


test_size


# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scaler = MinMaxScaler()


# In[18]:


scaler.fit(df_train)


# In[19]:


scaled_train = scaler.transform(df_train)
scaled_test = scaler.transform(df_test)


# In[20]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[21]:


length = 12


# In[22]:


generator = TimeseriesGenerator(scaled_train, scaled_train, length = length , batch_size=1)
val_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)


# In[23]:


scaled_train.shape


# In[24]:


n_features = scaled_train.shape[1]


# In[25]:


from tensorflow.keras.models import Sequential


# In[99]:


from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[100]:


model = Sequential()

model.add(LSTM(50, activation='relu',return_sequences=True, input_shape=(length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu',return_sequences=True, input_shape=(length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu', input_shape=(length, n_features)))
model.add(Dense(n_features))

model.compile(optimizer='adam', loss = 'mse')


# In[101]:


model.summary()


# In[102]:


from tensorflow.keras.callbacks import EarlyStopping


# In[103]:


early = EarlyStopping(monitor='val_loss', patience=2)


# In[104]:


model.fit(generator, 
          epochs=8, 
          validation_data=val_generator, 
          callbacks = [early]
         )


# In[105]:


loss = pd.DataFrame(model.history.history)


# In[106]:


loss.plot()


# In[107]:


test_predictions = []
n_features = scaled_train.shape[1]
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(df_test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[108]:


true_prediction = scaler.inverse_transform(test_predictions)


# In[109]:


true_prediction


# In[110]:


df_train.tail()


# In[111]:


ind = df_test.index


# In[ ]:





# In[112]:


true_prediction = pd.DataFrame(true_prediction,index=ind, columns=['P_Open', 'P_High', 'P_Low', 'P_Close', 'P_Volume'])


# In[113]:


true_prediction


# In[114]:


df_test


# In[115]:


final_df = pd.concat([df_test, true_prediction], axis=1)


# In[116]:


final_df


# In[117]:


final_df[['Open','P_Open']].plot()


# In[118]:


final_df[['Close','P_Close']].plot()


# In[119]:


final_df[['High','P_High']].plot()


# In[120]:


final_df[['Low','P_Low']].plot()


# In[121]:


final_df[['Volume','P_Volume']].plot()


# In[ ]:




