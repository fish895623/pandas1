# %%
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris


iris = load_iris()

print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])


# %%
test = [0, 50, 100]

# training data 준비과정
train_data = np.delete(iris.data, test, axis=0)
train_target = np.delete(iris.target, test)

# testing data 준비과정
test_data = iris.data[test]
test_target = iris.target[test]

# 결정트리 생성
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


print(test_target)
print(clf.predict(test_data))
