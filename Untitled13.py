# %%
from sklearn import tree


features = [[270, 0], [250, 0], [220, 1], [240, 1]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)


print(clf.predict([[245, 1]]))


# %%
