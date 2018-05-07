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
start_fit = time.time()
clf = clf.fit(X, y)
end_fit = time.time()
print("Time building forest => " , end_fit - start_fit)


num_q = int(input())

X_q = get_x(num_q)
y_q = get_y()
start_predict = time.time()
print("Score => ", clf.score(X_q, y_q))
end_predict = time.time()
print("Time predicting =>" , end_predict - start_predict)
