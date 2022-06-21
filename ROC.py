#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:28:34 2022

@author: Simone Ludwig
"""
from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
    PrecisionRecallDisplay
# f1_score, \ accuracy_score,
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns


plt.rcParams.update({'font.size': 8,
                     "figure.figsize": (7/2.54, 7/2.54)})
# for i in range(1, 13):
i = 6
step = "22k"
if i == 12:
    Confusion_Matrix = "Confusion Matrix of Training on Synthetic Data Step" + step
    Precision_Recall = "Precision Recall Curve of Training on Synthetic Data" + step
    ROCplot = "ROC of Training on Synthetic Data" + step
    Title = "Training on Synthetic Data"
else:
    Confusion_Matrix = "Confusion Matrix of Loop " + str(i) + " of Training on Real and Synthetic Data Step" + step
    Precision_Recall = "Precision Recall Curve of Loop " + str(i) + " of Training on Real and Synthetic Data Step" + step
    ROCplot = "ROC of Loop " + str(i) + " of Training on Real and Synthetic Data Step" + step
    Title = "Loop " + str(i)

df = pd.read_csv("ROC_" + str(i) + ".csv", header=0)
# print(df.shape)
actual_y = df.iloc[:, -1:]
probas_y = df.iloc[:, 1:-1]
y_true = df[["y_true"]]
y_pred = df[["y_pred"]]
# print(actual_y, "\n", probas_y, "\n", y_true, "\n", y_pred)

# plotting confusion matrix
conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(conf_matrix)
ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False,
                 xticklabels=True, yticklabels=True)
ax.set_title(Title)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
# Display the visualization of the Confusion Matrix.
plt.tight_layout()
plt.savefig(Confusion_Matrix + ".png", format="png", dpi=300)
plt.show()
plt.close()
plt.clf()

# plotting Precision-Recall curve
print('Precision: %.3f' % precision_score(y_pred, y_true))
print('Recall: %.3f' % recall_score(y_pred, y_true))
display = PrecisionRecallDisplay.from_predictions(y_true, y_pred, name="DNN model")
_ = display.ax_.set_title(Title)
plt.tight_layout()
plt.savefig(Precision_Recall + ".png", format="png", dpi=300)
plt.show()
plt.close()
plt.clf()

# plotting roc curve
skplt.metrics.plot_roc(y_true=actual_y, y_probas=probas_y,
                       title=Title, plot_micro=False,
                       plot_macro=False, text_fontsize=6.5)
# auc = skplt.metrics.roc_auc_score(actual_y, probas_y)
plt.tight_layout()
plt.savefig(ROCplot + ".png", format="png", dpi=300)
plt.show()
plt.close()
plt.clf()
