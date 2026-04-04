import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

ffnn_results = pd.read_csv('/Users/kleinjr1/Downloads/model3_preds_and_labels.csv')
xgboost_results = pd.read_csv('/Users/kleinjr1/Downloads/model_xgb_results.csv')
log_reg_results = pd.read_csv('/Users/kleinjr1/Downloads/model_results_logreg.csv')
rf_results = pd.read_csv('/Users/kleinjr1/Downloads/rf_model_results.csv')

model_results = {'FFNN': ffnn_results,
                 'XGBoost': xgboost_results,
                 'Logistic Regression': log_reg_results,
                 "Random Forest": rf_results}

fig, ax = plt.subplots()

for model_name, df in model_results.items():
    labels, preds = df['true_labels'], df['probabilities']
    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    avg = metrics.average_precision_score(labels, preds)
    ax.plot(recall, precision, label=f"{model_name} (Avg Prec: {avg:.3f})")

ax.legend()
ax.set_title("Precision-Recall for all models")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
plt.show()