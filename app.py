import pandas as pd

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import perceptron

# Load dataset
input_file = "blood-glucose-results.csv"
dataset = pd.read_csv(input_file, header = 0)

"""
# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('timestamp').size())


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
"""

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

"""
# Make predictions on validation dataset
lr = LinearRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_train)
print(X_train)
print (predictions)

rid = Ridge()
rid.fit(X_train, Y_train)
predictions = rid.predict(X_train)
print (predictions)

las = Lasso(alpha=0.1)
las.fit(X_train, Y_train)
predictions = las.predict(X_train)
print (predictions)

en = ElasticNet()
en.fit(X_train, Y_train)
predictions = en.predict(X_train)
print (predictions)

br = BayesianRidge()
br.fit(X_train, Y_train)
predictions = br.predict(X_train)
print (predictions)
"""

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('RID', Ridge()))
models.append(('LAS', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('BR', BayesianRidge()))

# Test options and evaluation metric
seed = 7
scoring = 'mean_squared_error'

# evaluate each model in turn
results = []
names = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train, Y_train)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
