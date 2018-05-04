from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.datasets import make_classification
import graphviz 
import time

num_trees = int(input())
max_depth = int(input())
num_data = int(input())
len_data = int(input())

def get_x(num_data):
    X = []
    for _ in range(int(num_data)):
        aux = input()
        aux = aux.split()
        aux = [float(x) for x in aux]
        X.append(aux)
    return X

def get_y():
    y = input()
    y = y.split()
    y = [float(x) for x in y]
    return y



X = get_x(num_data)
y = get_y()

clf = RandomForestClassifier(max_depth=max_depth, random_state=None, n_jobs=-1, n_estimators=num_trees)
clf = clf.fit(X, y)

num_q = int(input())

X_q = get_x(num_q)
y_q = get_y()
start = time.time()
print(clf.score(X_q, y_q))
end = time.time()
print(end - start)
