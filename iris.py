
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

x = dataset.values[:, 0:4]
y = dataset.values[:, 4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state = 1)

#Trying different models, looking for the best fit via K-Fold X-Validation
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(("Linear Disc. Analysis", LinearDiscriminantAnalysis()))
models.append(("K-Nearest Neighbors", KNeighborsClassifier()))
models.append(("Class + Regression Trees", DecisionTreeClassifier()))
models.append(("Naive Bayes", GaussianNB()))
models.append(("Support Vector Machines", SVC(gamma='auto')))

results = []
model_names = []
for (name, model) in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results.mean())
	model_names.append(name)

#Choosing best model via highest mean score
chosen_model_index = results.index(max(results))
print('Chosen Model: %s' % (model_names[chosen_model_index]))
print()

#Uncomment for graph of accuracy results for each model
#pyplot.boxplot(results, labels=names)
#pyplot.show()

#Makin' preditions
model = models[chosen_model_index][1]
model.fit(x_train, y_train)
results = model.predict(x_validation)
i = 0
print("Predictions")
while i < len(results):
	print(x_validation[i], ": ", results[i])
	i += 1

#How'd it go?
print()
print("Accuracy")
print(accuracy_score(y_validation, results))
print(confusion_matrix(y_validation, results))
print(classification_report(y_validation, results))