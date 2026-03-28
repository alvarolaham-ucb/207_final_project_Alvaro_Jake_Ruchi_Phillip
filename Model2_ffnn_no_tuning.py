import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
from utils import model_builder_factory, build_binary_classifier
import matplotlib.pyplot as plt

# Parameters
analysis_df_path = '/Users/kleinjr1/Downloads/analysis_df.csv' # output from eda_and_preprocessing ruchit branch
output_fig_path =  '/Users/kleinjr1/Downloads/output.png'
# Load df
df = pd.read_csv(analysis_df_path)

# Set up data variables/splits
Y = df[['fire_start_day']]
temp = df.copy()
temp = temp.drop(columns=['fire_start_day'])
X = temp
X_temporary, X_test, Y_temporary, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_temporary, Y_temporary, test_size=0.25, random_state=0)

###########
# Data prep
############

# Scale features
numeric_cols = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED', 'LAGGED_PRECIPITATION', 'LAGGED_AVG_WIND_SPEED']
numeric_cols_lower = [i.lower() for i in numeric_cols]
scaler = StandardScaler()

X_train_std = X_train.copy()
X_val_std = X_val.copy()
X_test_std = X_test.copy()

X_train_std[numeric_cols_lower] = scaler.fit_transform(X_train[numeric_cols_lower])
X_val_std[numeric_cols_lower] = scaler.transform(X_val[numeric_cols_lower])
X_test_std[numeric_cols_lower] = scaler.transform(X_test[numeric_cols_lower])

# Class imbalance
#TODO

# Build the model
model = build_binary_classifier(input_dim=(X_train.shape[1],),
                                hidden_layer_sizes=[5, 5])


history = model.fit(X_train_std,
                    Y_train,
                    batch_size=32,
                    validation_data=(X_val_std, Y_val),
                    class_weight={0: 1, 1: 2}, # Adjusting for the 67/33 split
                    epochs=10)

# Weights and biases
W_final_layer, b_final_layer = model.layers[-1].get_weights()
# for layer in model.layers:
#     W, b = layer.get_weights()

train_loss = history.history['loss'][-1]
train_binary_accuracy = history.history['binary_accuracy'][-1]
val_loss = history.history['val_loss'][-1]
val_binary_accuracy = history.history['val_binary_accuracy'][-1]

percentage_diff = ((train_loss - val_loss)/train_loss)*100

# Plot
history = history.history


fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history['loss'], lw=2, color='darkgoldenrod')
plt.plot(history['val_loss'], lw=2, color='indianred')
plt.legend(['Train', 'Validation'], fontsize=10)
ax.set_xlabel('Epochs', size=10)
ax.set_ylabel('Loss')
ax.set_title('Loss for train and validation')

# plot accuracy for train and validation
ax = fig.add_subplot(1, 2, 2)
plt.plot(history['binary_accuracy'], lw=2, color='darkgoldenrod')
plt.plot(history['val_binary_accuracy'], lw=2, color='indianred')
plt.legend(['Train', 'Validation'], fontsize=10)
ax.set_xlabel('Epochs', size=10)
ax.set_ylabel('Binary Accuracy')
ax.set_title('Binary Accuracy for train and validation')
plt.savefig(output_fig_path)