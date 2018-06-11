import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn_pandas import DataFrameMapper, cross_val_score

from plotting import plot_learning_curve

## Get data
dataset = pandas.read_csv('Data/stats2.csv')


# To determine how many samples we want.
n = 3000

# X = data and y = target
array = dataset.values
X = array[:n,1:]
y = array[:n,0]

## Initialization of our data
validation_size = 0.2
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


## Debugging issues
print("Classifier: %s" % dataset.columns.values[0])
print("Attributes: %s" % dataset.columns.values[1:])


## Create accuracies for the models.
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

# For loop for each model we want.
for name, model in models:

	## Finding accuracy
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

	# Now for graphs
	title = "Learning Curves (" + name + ")"
	print("Creating Learning curves for %s" % name)

	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

	# Get plot for this model and data.
	plot_learning_curve(model, title, X, y, ylim=(0.3, 1.01), cv=cv, n_jobs=1)

	plt.show()
