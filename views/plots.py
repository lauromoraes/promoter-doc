import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')

def plot_roc(fp_rate, tp_rate):
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(fp_rate, tp_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

def plot_confusion(y_test, y_pred_test):
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['Prom', 'Non-Prom']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()