from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.datasets import make_classification
import graphviz 

# X, y = make_classification(n_samples=10, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

# print(len(X[0:79]))
# print(len(X[0]))
# print(X)
# print(" ".join(str(x) for x in X[:799]))
# print(y)
num_trees = int(input())
max_depth = int(input())
num_data = int(input())
len_data = int(input())

X = []
for _ in range(int(num_data)):
    aux = input()
    aux = aux.split()
    aux = [float(x) for x in aux]
    X.append(aux)
    # print(aux)
# print(X)
y = input()
y = y.split()
y = [float(x) for x in y]
# print(y)
# for _ in range(int(nu))

clf = RandomForestClassifier(max_depth=max_depth, random_state=None)
clf = clf.fit(X, y)
print(clf.predict_proba([[842, 0,  2.2, 0,    1, 0, 7,  0.6, 188, 2,
                           2,   20, 756, 2549, 9, 7, 19, 0,   0,   1]]))
# print(clf)
print(clf.feature_importances_)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("test")