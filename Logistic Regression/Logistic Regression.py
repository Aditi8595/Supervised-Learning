import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']


# load dataset
pima = pd.read_csv("diabetes.csv", header=1, names=col_names)
print(pima.head())

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

#################################################


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


###########################################



# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

##############################################
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
###########################################


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
"""data – 2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.
annot – an array of same shape as data which is used to annotate the heatmap.
cmap – a matplotlib colormap name or object. This maps the data values to the color space.
fmt – string formatting code to use when adding annotations.
linewidths – sets the width of the lines that will divide each cell."""

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
#position : {'top', 'bottom'}

plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

"""Precision: Precision is about being precise, i.e.,
how accurate your model is. In other words, you can say,
when a model makes a prediction, how often it is correct.
In your prediction case, when your Logistic Regression model
predicted patients are going to suffer from diabetes,
that patients have 76% of the time.

Recall: If there are patients who have diabetes in the test set
and your Logistic Regression model can identify it 58% of the time.

"""
"""Receiver Operating Characteristic(ROC) curve is a plot
of the true positive rate against the false positive rate.
It shows the tradeoff between sensitivity and specificity."""

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

"""AUC score for the case is 0.86. AUC score 1 represents perfect classifier,
and 0.5 represents a worthless classifier."""








