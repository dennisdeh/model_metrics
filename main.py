"""
This is an example using the model performance metrics of this module
in some simple supervised learning workflow.
"""
import numpy as np
from model_metrics.metrics import model_performance_metrics
from sklearn.datasets import load_wine, load_breast_cancer
from scipy import stats
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#%% Example 1: binary model
"""
Data: breast_cancer dataset (2 classes; pre-processing necessary)
Model: LogisticRegression
"""
# 1: pre-process data
# 1.1: load data set
X, y = load_breast_cancer(return_X_y=True)
# 1.2: data pre-processing
# 1.2.1: inspect data set structure (all features are numerical)
print(f"Number of missing values: {np.isnan(X).sum()}")  # = 0; no imputation needed
print(stats.describe(X))  # skewness and kurtosis present in some of the features
"""
Conclusion of analysis:
Data needs to be normalised and outliers removed.
We will do a winsorisation to treat outliers, and then standardise the data
"""
# 1.2.1: transform data
# 99% winsorisation
X_win = stats.mstats.winsorize(X, limits=[0.01, 0.01], axis=0).data
# standardise the data; remove mean and rescale
s = StandardScaler()
Xtf = s.fit_transform(X)
# 1.3: create training and validation samples
X_train, X_val, y_train, y_val = train_test_split(Xtf, y, test_size=0.2, random_state=1)

# 2: construct model
# 2.1: define model and fit it
model = LogisticRegression()
model = model.fit(X_train, y_train)

# 3: evaluate performance
# 3.1: get prediction on the validation sample
y_pred_val = model.predict(X_val)
y_pred_proba_val = model.predict_proba(X_val)
# 3.2: calculate model performance metrics
dict_metrics = model_performance_metrics(y_true=y_val,
                                         y_pred_labels=y_pred_val,
                                         y_pred_probabilities=y_pred_proba_val,
                                         y_train=y_train,
                                         target_type="binary")
# 3.3: conclusion
"""
The model performance metrics gives insight to different aspects of how well the model performs.
The model is a good classifier with high accuracy (metrics to look at: accuracy, 
confusion matrix (almost diagonal), F1 metrics). The precision and recall are also high,
The data is not severely imbalance and thus the high AUC metrics shows that the model has a 
high true positive rate at even low false positive rates.
"""


#%% Example 2: multiclass model
"""
Data: wine dataset (3 classes; no additional data preparation needed)
Model: DecisionTreeClassifier
"""
# 1: pre-process data
# 1.1: load data set
X, y = load_wine(return_X_y=True)
# 1.2: data pre-processing
# no further data pre-processing is really needed here since we use a decision tree
# 1.3: create training and validation samples
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# 2: modelling step
# 2.1: define model and fit it
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# 2.2: visually inspect final tree structure
tree.plot_tree(clf)
plt.show()

# 3: evaluate performance
# 3.1: get prediction on the validation sample
y_pred_val = clf.predict(X_val)
y_pred_proba_val = clf.predict_proba(X_val)
# 3.2: calculate model performance metrics
dict_metrics = model_performance_metrics(y_true=y_val,
                                         y_pred_labels=y_pred_val,
                                         y_pred_probabilities=y_pred_proba_val,
                                         y_train=y_train,
                                         target_type="multiclass")
# 3.3: conclusion
"""
The model performance metrics gives insight to different aspects of how well the model performs.
The model turns out to be a good classifier with high accuracy (metrics to look at: accuracy, 
confusion matrix (almost diagonal), F1 metrics).
Furthermore, the high AUC metrics shows that the model has a high true positive rate at even low false 
positive rates. The relative lower one-vs-one AUC values suggests that the model has minor issues
distinguishing between some pairs of target classes.
The precision and recall metrics are relatively high, suggesting that the model classifies well
the relevant classes, despite the imbalance in number of observations in each class in the
validation data.
"""
