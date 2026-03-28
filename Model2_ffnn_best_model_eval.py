from utils import model_builder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
import keras
import matplotlib.pyplot as plt
import numpy as np
import json
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


# Parameters
analysis_df_path = '/Users/kleinjr1/Downloads/analysis_df.csv' # output from eda_and_preprocessing ruchit branch
input_params_path = '/Users/kleinjr1/Downloads/best_hps.json'

# Load df
df = pd.read_csv(analysis_df_path)
# Load hyperparams
with open(input_params_path, "r") as f:
    hp_dict = json.load(f)

# Recreate best_hps object
hp = HyperParameters()
hp.values.update(hp_dict)

# Do data setup
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

# Recreate model
# model = model_builder(hp, (X_train.shape[1],))
#
#
# history = model.fit(X_train_std,
#                     Y_train,
#                     batch_size=32,
#                     validation_data=(X_val_std, Y_val),
#                     class_weight={0: 1, 1: 2}, # Adjusting for the 67/33 split
#                     epochs=100)
#
# model.save("/Users/kleinjr1/Downloads/best_model.keras", save_format="tf")
# history = history.history
# with open("/Users/kleinjr1/Downloads/best_model_history.json", "w") as f:
#     json.dump(history.history, f)

# # Load saved model
model = keras.models.load_model("/Users/kleinjr1/Downloads/best_model.keras")

# Load saved history
with open("/Users/kleinjr1/Downloads/best_model_history.json", 'r') as file:
    history = json.load(file)

# Show learned model with weights and biases
W_final_layer, b_final_layer = model.layers[-1].get_weights()

# Get predictions
train_preds_probs = model.predict(X_train_std).flatten()
train_preds = (train_preds_probs >= 0.5).astype(int)
val_preds_probs = model.predict(X_val_std).flatten()
val_preds = (val_preds_probs >= 0.5).astype(int)
test_preds_probs = model.predict(X_test_std).flatten()
test_preds = (test_preds_probs >= 0.5).astype(int)


# Calculate accuracies manually
final_model_train_accuracy_df = pd.DataFrame({'predictions': train_preds, 'actuals': Y_train.values.astype(int).flatten()})
final_model_val_accuracy_df = pd.DataFrame({'predictions': val_preds, 'actuals': Y_val.values.astype(int).flatten()})
final_model_test_accuracy_df = pd.DataFrame({'predictions': test_preds, 'actuals': Y_test.values.astype(int).flatten()})

agg_accuracy_df = pd.concat([final_model_train_accuracy_df, final_model_val_accuracy_df, final_model_test_accuracy_df], axis=0)

train_accuracy = np.sum(final_model_train_accuracy_df['predictions'] == final_model_train_accuracy_df['actuals']) / len(final_model_train_accuracy_df)
val_accuracy = np.sum(final_model_val_accuracy_df['predictions'] == final_model_val_accuracy_df['actuals']) / len(final_model_val_accuracy_df)
test_accuracy = np.sum(final_model_test_accuracy_df['predictions'] == final_model_test_accuracy_df['actuals']) / len(final_model_test_accuracy_df)

agg_accuracy = np.sum(agg_accuracy_df['predictions'] == agg_accuracy_df['actuals']) / len(agg_accuracy_df)

# Optional to get accuracies from model.evaluate as well (returned as 2nd param given how model_builder is set up)
print("Gap between aggregate test and train accuracy", test_accuracy - train_accuracy)



# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(Y_test, test_preds, cmap=plt.cm.Blues,
                                              display_labels=["No Fire [0]", "Fire [1]"])

disp.ax_.set_title("Confusion Matrix (using test data)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
# the precision and recall

print(classification_report(
    y_true = Y_test,
    y_pred = test_preds,  # labels predicted by model_tf
    target_names=['No Fire', 'Fire']))


# Training Process diagnostics
output_fig_path = '/Users/kleinjr1/Downloads/results.png'

val_acc_per_epoch = history['val_binary_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Get final losses
train_loss = history['loss'][-1]
val_loss = history['val_loss'][-1]

print('Final train loss:', train_loss)
print('Final validation loss:', val_loss)


percentage_diff = ((train_loss - val_loss)/train_loss)*100
print(" percentage difference between the final losses observed on the training and validation datasets:",percentage_diff )



# plot loss for train and validation
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history['loss'], lw=2, color='darkgoldenrod')
plt.plot(history['val_loss'], lw=2, color='indianred')
plt.legend(['Train', 'Validation'], fontsize=10)
ax.set_xlabel('Epochs', size=10)
ax.set_ylabel('Loss')
ax.set_title('Loss for train and validation');

# plot accuracy for train and validation
ax = fig.add_subplot(1, 2, 2)
plt.plot(history['binary_accuracy'], lw=2, color='darkgoldenrod')
plt.plot(history['val_binary_accuracy'], lw=2, color='indianred')
plt.legend(['Train', 'Validation'], fontsize=10)
ax.set_xlabel('Epochs', size=10)
ax.set_ylabel('Binary Accuracy')
ax.set_title('Binary Accuracy for train and validation')

plt.savefig(output_fig_path)

# Overall loss for the model
train_results = model.evaluate(X_train_std, Y_train, verbose=0)
test_results = model.evaluate(X_test_std, Y_test, verbose=0)

print("Overall model train loss:", train_results[0])
print("Overall model test loss:", test_results[0])
print("Gap between train and test loss:", test_results[0] - train_results[0])

# AUROC
fpr, tpr, thresholds = roc_curve(Y_test, test_preds_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (using test data)")
plt.legend()
plt.show()

# AUPR
precision, recall, thresholds = precision_recall_curve(Y_test, test_preds_probs)
pr_auc = average_precision_score(Y_test, test_preds_probs)

plt.figure()
plt.plot(recall, precision, label=f"Avg Prec = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (using test data)")
plt.legend()
plt.show()

print('done')