# %%
from sklearn import tree


features = [[270, 0], [250, 0], [220, 1], [240, 1]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)


print(clf.predict([[245, 1]]))


# %%
from sklearn.datasets import load_iris


iris = load_iris()

print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

# %%
