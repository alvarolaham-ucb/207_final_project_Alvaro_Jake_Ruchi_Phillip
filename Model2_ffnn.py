import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
from keras import metrics
from utils import model_builder_factory
import matplotlib.pyplot as plt
import json

# Parameters
analysis_df_path = '/Users/kleinjr1/Downloads/analysis_df2.csv' # output from eda_and_preprocessing ruchit branch
output_params_path = '/Users/kleinjr1/Downloads/best_hps3.json'

# Load df
df = pd.read_csv(analysis_df_path)

# Set up data variables/splits
Y = df[['fire_start_day']]
temp = df.copy()
temp = temp.drop(columns=['fire_start_day'])
X = temp
X_temporary, X_test, Y_temporary, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_temporary, Y_temporary, test_size=0.25, random_state=0, stratify=Y_temporary)

###########
# Data prep
############

# Scale features
numeric_cols = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED', 'LAGGED_PRECIPITATION', 'LAGGED_AVG_WIND_SPEED', 'DAYS_WITHOUT_RAIN']
numeric_cols_lower = [i.lower() for i in numeric_cols]
scaler = StandardScaler()

X_train_std = X_train.copy()
X_val_std = X_val.copy()
X_test_std = X_test.copy()

X_train_std[numeric_cols_lower] = scaler.fit_transform(X_train[numeric_cols_lower])
X_val_std[numeric_cols_lower] = scaler.transform(X_val[numeric_cols_lower])
X_test_std[numeric_cols_lower] = scaler.transform(X_test[numeric_cols_lower])


# Hyperparameter tuning
# Tuner
tuner = kt.Hyperband(
    hypermodel=model_builder_factory(input_dim=(X_train.shape[1],)), # passing that function from above telling tuner how to build the model_tf
    objective='val_binary_accuracy',
    max_epochs=100,
    factor=3,
    directory='tuning_dir',
    project_name='lr_batch_tuner',
    overwrite=True
)

# Early stop condition: stop once reach a certain val loss
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Actual HP search
tuner.search(X_train_std,
             Y_train,
             batch_size=32,
             validation_data=(X_val_std, Y_val),
             class_weight={0: 1, 1: 2},
             epochs=100,
             callbacks = [stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
with open(output_params_path, "w") as f:
    json.dump(best_hps.values, f, indent=2)
#
# # Build the model with the optimal hyperparameters
# model = tuner.hypermodel.build(best_hps)
#
# # Find best epoch
# history = model.fit(X_train_std,
#                     Y_train,
#                     batch_size=32,
#                     validation_data=(X_val_std, Y_val),
#                     class_weight={0: 1, 1: 2}, # Adjusting for the 67/33 split
#                     epochs=100)
#
# val_acc_per_epoch = history.history['val_binary_accuracy']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))
#
# # Show learned model with weights and biases from
# W_final_layer, b_final_layer = model.layers[-1].get_weights()
#
# train_loss = history.history['loss'][-1]
# train_binary_accuracy = history.history['binary_accuracy'][-1]
# val_loss = history.history['val_loss'][-1]
# val_binary_accuracy = history.history['val_binary_accuracy'][-1]
#
# print('Final train loss:', train_loss)
# print('Final validation loss:', val_loss)
# print('Final train_accuracy:', train_binary_accuracy)
# print("Final_val_accuracy:", val_binary_accuracy)
#
# percentage_diff = ((train_loss - val_loss)/train_loss)*100
# print(" percentage difference between the losses observed on the training and validation datasets:",percentage_diff )
#
# # grab history
# history = history.history
#
# # plot loss for train and validation
# fig = plt.figure(figsize=(16, 4))
# ax = fig.add_subplot(1, 2, 1)
# plt.plot(history['loss'], lw=2, color='darkgoldenrod')
# plt.plot(history['val_loss'], lw=2, color='indianred')
# plt.legend(['Train', 'Validation'], fontsize=10)
# ax.set_xlabel('Epochs', size=10)
# ax.set_ylabel('Loss')
# ax.set_title('Loss for train and validation');
#
# # plot accuracy for train and validation
# ax = fig.add_subplot(1, 2, 2)
# plt.plot(history['binary_accuracy'], lw=2, color='darkgoldenrod')
# plt.plot(history['val_binary_accuracy'], lw=2, color='indianred')
# plt.legend(['Train', 'Validation'], fontsize=10)
# ax.set_xlabel('Epochs', size=10)
# ax.set_ylabel('Binary Accuracy')
# ax.set_title('Binary Accuracy for train and validation')
#
# # Save artifacts
# # Plot
# plt.savefig(output_fig_path)


