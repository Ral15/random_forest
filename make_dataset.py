# from sklearn.datasets import make_classification
# from sklearn import tree
# from sklearn.datasets import load_iris
# import graphviz 

import random

lines = []
with open('train.in', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(",")
        # print(line[:-1])
        lines.append(line)

random.shuffle(lines)

train_data = lines[0:1399] #1400
test_data = lines[1400:] # 600

print(len(train_data))
print(len(train_data[0][:-1]))


train_target = []
for l in train_data:
    print(" ".join(str(x) for x in l[:-1]))
    train_target.append(l[-1])
# print(train_target)

# for l in train_target:
print(" ".join(str(x) for x in train_target))

print(len(test_data))
test_target = []
for l in test_data:
    print(" ".join(str(x) for x in l[:-1]))
    test_target.append(l[-1])

# for l in test_target:
print(" ".join(str(x) for x in test_target))

# print(random.shuffle(lines))
# print("\n".join(str(x) for x in lines))
# target = []
# for l in lines[0:1399]:
# #     # print(l)
#     target.append(l[-1])
# print(lines)
# for l in lines:
#     # l.split("\n")
#     target.append(l[-1])

# print(target)