# from sklearn.datasets import make_classification
# from sklearn import tree
# from sklearn.datasets import load_iris
# import graphviz 

# # samples = 20
# # features = 4


# # X, y = make_classification(n_samples=samples, n_features=features, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
# # print(samples)
# # print(features)
# # print(" \n".join(str(x) for x in X))
# # print(y)


# iris = load_iris()
# data_len = len(iris.data)
# features_len = len(iris.data[0])
# print(data_len)
# print(features_len)
# print(" \n".join(str(x) for x in iris.data))
# print(iris.target)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
# # print(clf)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
# # print(clf.decision_path(iris.data))

lines = []
with open('train.in', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(",")
        print(line[:-1])
        lines.append(line)
# print(lines)
# print("\n".join(str(x) for x in lines))
target = []
for l in lines:
    # print(l)
    target.append(l[-1])
# print(lines)
# for l in lines:
#     # l.split("\n")
#     target.append(l[-1])

print(target)