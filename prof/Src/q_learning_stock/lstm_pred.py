# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import util
import indicators
import os

def lstm_model():
    model = tf.keras.Sequential()
    #############
    # model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(lookback, feature_count), return_sequences=True))
    # # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64))
    # model.add(tf.keras.layers.Dense(16, kernel_initializer='uniform', activation='relu'))
    # model.add(tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='linear'))#, activation='linear'))
    #############
    # model.add(tf.compat.v1.keras.layers.CuDNNLSTM(100, input_shape=(lookback, feature_count), kernel_initializer='random_uniform'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(20,activation='relu'))
    # model.add(tf.keras.layers.Dense(1,activation='linear'))
    #############
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32, input_shape=(lookback, feature_count)))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(lr=0.001) #RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def create_dataset(data,lookback):
    """process the data into n day look back slices
    """
    X,Y = [],[]
    for i in range(len(data)-lookback-1):
        # X_data = np.concatenate((data[:, :3], data[:, 4:]), axis=1)
        X.append(data[i:(i+lookback), 1:])
        Y.append(data[(i+lookback), 0])
    return np.array(X),np.array(Y)

# df = util.read_df('data/indexes/^BVSP.csv')
df = pd.read_csv('data/indexes/^TWII.csv', usecols=['Date','Close'], parse_dates=['Date'])
df = df[df['Close'] != 0]
# df = util.add_features_to_df(df)
df['EMA'] = indicators.exponential_moving_avg(df, window_size=10, center=False)
df['MACD_Line'] = indicators.macd_line(df, ema1_window_size=10, ema2_window_size=20, center=False)
df['MACD_Signal'] = indicators.macd_signal(df, window_size=10, ema1_window_size=10, ema2_window_size=20, center=False)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.index = df['Date']
df.drop('Date', axis=1, inplace=True)
feature_count = df.values.shape[1] - 1
df.head(5)

#%%
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(df.values)
print(X.shape)
# 80:20 split
train_split_len = int(len(df)/10*8)
train = X[:train_split_len, :]
test = X[train_split_len:, :]
lookback = 7

train_x, train_y = create_dataset(train, lookback)
test_x, test_y = create_dataset(test, lookback)
print(train_x.shape)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], feature_count))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], feature_count))

#%%

# tensorboard = tf.keras.callbacks.TensorBoard(log_dir="data\\log\\test1")
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.0001)
# mcpt = tf.keras.callbacks.ModelCheckpoint('data\\metadata\\best_lstm.h5', monitor='val_loss', verbose=1,
#         save_best_only=True, save_weights_only=False, mode='min', period=1)

model = lstm_model()

# history = model.fit(
#     train_x, train_y,
#     epochs=200,
#     verbose=1,
#     batch_size=16,
#     validation_data=(test_x,test_y),
#     shuffle=False,
#     # callbacks=[es]
# )
history = model.fit(train_x,train_y,
                    epochs=200, verbose=0, batch_size=16,
                    shuffle=False, validation_data=(test_x, test_y))

#%%
# Save model and weights
with open("data/rl/index/stock_pred_^TWII.json", "w") as json_file:
    json_file.write(model.to_json())
model.save('data/rl/index/stock_pred_^TWII.hdf5')

#%%
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend()

#%%


print(test_x.shape)
print(test_x)
print(test_y.shape)
Xt = model.predict(test_x)
print(Xt.shape)
# print(test_y)
temp_x = test_x[:,0,:]
# temp_data = np.concatenate((temp_x[:,:3], Xt, temp_x[:,3:]), axis=1)
print(temp_x.shape)
temp_data = np.concatenate((Xt, temp_x), axis=1)
print(temp_data.shape)
# temp_data2 = np.concatenate((temp_x[:,:3], test_y.reshape(len(test_y), 1), temp_x[:,3:]), axis=1)
temp_data2 = np.concatenate((test_y.reshape(len(test_y), 1), temp_x), axis=1)
print(temp_data2.shape)
plt.plot(scaler.inverse_transform(temp_data2)[:, 0], label='Actual')
plt.plot(scaler.inverse_transform(temp_data)[:, 0], label='Predicted')
plt.legend()

#%%
# model = tf.keras.models.load_model('data/rl/index/stock_pred.hdf5')
