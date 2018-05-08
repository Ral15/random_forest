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
num_q = int(input())
X_q = get_x(num_q)
y_q = get_y()

results_fit = []
results_score = []
results_predict =[]

# ranges = [x + 25 for x in range(25, 501)]
# print(ranges)
for i in range(25, 501, 25):
    clf = RandomForestClassifier(max_depth=10, random_state=None, n_jobs=-1, n_estimators=i, bootstrap=False)
    start_fit = time.time()
    clf = clf.fit(X, y)
    end_fit = time.time()
    # print("Time building forest => " , end_fit - start_fit)
    results_fit.append((i,  end_fit - start_fit))

    
    start_predict = time.time()
    results_score.append((i, clf.score(X_q, y_q)))
    # clf.score(X_q, y_q)
    end_predict = time.time()
    results_predict.append((i,  end_predict - start_predict))
    # print("Time predicting =>" , end_predict - start_predict)

print(results_fit)
print(results_score)
print(results_predict)