# %%
# Importing window generator - The WindowGenerator Class has been define in another py file, so it will be imported from there
from time_series_helper import WindowGenerator

# %%
# Import necessary modules

import os
import IPython
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import IPython.display
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
# Read the file
df = pd.read_csv('cases_malaysia.csv')

# %%
# Data Inspection for the df (do some data cleaning if necessary)
df.head()
# %%
df.info()
# %%
df.isnull().sum()
# %%
df = df.fillna(0)
# %%
# Starndardize all date in date column
Date = pd.to_datetime(df.pop('date'))
# %%
# Convert all columns to int64
df = df.astype('int64')

# %%
# Checking the plot for new_cases
plot_cols = ['cases_new']
plot_features = df[plot_cols]
plot_features.index = Date
_ = plot_features.plot(subplots=True)

# %%
# Data cleaning has been done, now there is no more null values, and the datatype is same for all columns
# Data Preprocessing
# Splitting the data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.6)]
val_df = df[int(n*0.6):int(n*0.8)]
test_df = df[int(n*0.8):]

num_features = df.shape[1]

# %%
# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
# initialize window for lstm
window_1 = WindowGenerator(
    input_width=30, label_width=30, shift=1, 
    train_df=train_df, val_df=val_df,
    test_df=test_df, 
    label_columns=['cases_new'])

window_1.plot(plot_col='cases_new')
window_1

# %%
# For tensorboard
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = tf.keras.callbacks.TensorBoard(logpath)

# %%
MAX_EPOCHS = 40
# Function for compile and fit for training
def compile_and_fit(model, window, patience=3):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
  patience=patience,
  mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping,tb])

# %%
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(64,return_sequences=True))
lstm_model.add(tf.keras.layers.Dropout(0.2))
lstm_model.add(tf.keras.layers.Dense(1))

# %%
#Train the model
history_model = compile_and_fit(lstm_model, window_1)

# %%
#get MAPE test
MAPE_test = lstm_model.evaluate(window_1.test)

# %%
# Prediction plot for single step with label
window_1.plot(model=lstm_model, plot_col='cases_new')

# %%
# Get the model architecture
from tensorflow.keras.utils import plot_model

plot_model(lstm_model, to_file='model_architecture.png',show_shapes=True)

# %%
# Make predictions on the test set
predictions = lstm_model.predict(window_1.test)
test_true = np.concatenate([y for x, y in window_1.test], axis=0)

# Inverse transform to the original scale
predictions = predictions * train_std['cases_new'] + train_mean['cases_new']
test_true = test_true * train_std['cases_new'] + train_mean['cases_new']

# Plotting actual and predicted cases
plt.figure(figsize=(10, 6))
plt.plot(test_true[:, 0], label='Actual Cases',alpha=0.5)
plt.plot(predictions[:, 0], label='Predicted Cases')
plt.title('Actual vs Predicted Cases')
plt.xlabel('Time Steps')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()
# %%
