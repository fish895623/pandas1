# %%
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


iris = load_iris()
sepal = iris.data[:, 0:2]
kind = iris.target
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.plot(sepal[kind == 0][:, 0],
         sepal[kind == 0][:, 1], "ro", label='Setosa')
plt.plot(sepal[kind == 1][:, 0],
         sepal[kind == 1][:, 1], "bo", label='Versicolor')
plt.plot(sepal[kind == 2][:, 0],
         sepal[kind == 2][:, 1], "yo", label='Virginica')
plt.legend()

# %% p264
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print('예측:\n', predictions)
print('정답(y_test):\n', y_test)

# %% 오차율 확인

print(accuracy_score(y_test, predictions))
