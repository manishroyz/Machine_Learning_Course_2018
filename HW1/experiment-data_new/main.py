from data import Data
import numpy as np
from sklearn import preprocessing
from dt import DecisionTreeClassifier

# FUNCTIONS
def encode_array(arr):
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    arr = np.apply_along_axis(le.fit_transform, 0, arr)
    labels = arr[:,0].tolist()
    arr = ohe.fit_transform(arr[:,1:]).toarray()
    arr = np.hstack((np.array(labels)[:, np.newaxis], arr))
    return arr

def accuracy(output, labels):
    count = 0
    for i, o in enumerate(output):
        if o == labels[i]:
            count = count + 1

    return count / len(output)

def test_tree(train, test, depth):
    dt = DecisionTreeClassifier()
    dt.fit(train, depth)
    output = dt.predict(test)
    acc = accuracy(output, test[:,0])
    return acc

# Data objects for Train & Test data sets
train_fpath = ".//data_new//train.csv"
test_fpath = ".//data_new//test.csv"

print("Calling  data.py ....")
train_obj = Data(fpath=train_fpath)
train = train_obj.raw_data
print("Raw Ytrain dataset....")
print(train)
test = Data(fpath = test_fpath).raw_data


# print(type(data_obj.raw_data))
# print(data_obj.index_column_dict)
# print(data_obj.column_index_dict)
# print(data_obj.raw_data)

# for obj in data_obj.attributes:
#     print(data_obj.attributes[obj].__repr__)
#
# temp = encode_array(train)
# labels = temp[:, 0]
# dt = DecisionTreeClassifier()
# dt.fit(temp, 10)
# output = dt.predict(temp)
#

train = encode_array(train)
test = encode_array(test)

print("Train data after encoding....")
print(train)

print(".........................Train data.......................")
print("Varying depths and testing....")
for i in range(2, 22):
    a_train = test_tree(train, train, i)
    print(i, a_train)

print(".........................Test data........................")
print("Varying depths and testing....")
for i in range(2, 22):
    a_test = test_tree(train, test, i)
    print(i, a_test)


# #######################################################################################
# Section to perform  cross-validation

def cv(train, test):
    train = encode_array(train)
    test = encode_array(test)
    acc_map = []
    for d in depth:
        acc = test_tree(train, test, d)
        acc_map.append(acc)
    return acc_map

def cv_for_depth():
    data_obj1 = Data(fpath = file1).raw_data
    data_obj2 = Data(fpath = file2).raw_data
    data_obj3 = Data(fpath = file3).raw_data
    data_obj4 = Data(fpath = file4).raw_data
    data_obj5 = Data(fpath = file5).raw_data

    print(data_obj1.shape)
    print(data_obj2.shape)
    print(data_obj3.shape)
    print(data_obj4.shape)
    print(data_obj5.shape)

    train1 = np.vstack((data_obj1, data_obj2, data_obj3, data_obj4))
    test1 = data_obj5
    train2 = np.vstack((data_obj1, data_obj2, data_obj3, data_obj5))
    test2 = data_obj4
    train3 = np.vstack((data_obj1, data_obj2, data_obj5, data_obj4))
    test3 = data_obj3
    train4 = np.vstack((data_obj1, data_obj5, data_obj3, data_obj4))
    test4 = data_obj2
    train5 = np.vstack((data_obj5, data_obj2, data_obj3, data_obj4))
    test5 = data_obj1

    print("train1 .....train5")
    print(train1)
    print(train2)
    print(train3)
    print(train4)
    print(train5)

    a1 = cv(train1,test1)
    a2 = cv(train2,test2)
    a3 = cv(train3,test3)
    a4 = cv(train4,test4)
    a5 = cv(train5,test5)

    print("a1 a2......a5")
    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)

    a = np.vstack((a1, a2, a3, a4, a5)).T
    a_final = np.mean(a, 1)
    print("a....")
    print(a)
    print("a_final...")
    print(a_final)

    return [a_final, depth]

file1 = ".//data_new//CVfolds_new//fold1.csv"
file2 = ".//data_new//CVfolds_new//fold2.csv"
file3 = ".//data_new//CVfolds_new//fold3.csv"
file4 = ".//data_new//CVfolds_new//fold4.csv"
file5 = ".//data_new//CVfolds_new//fold5.csv"

depth = [1, 2, 3, 4, 5, 10, 15]
#
# print("Calling cross-validate()....")
# [a_final, depth] = cv_for_depth()
# print("As computed the best accuracy is for depth = 4")
# print("Testing for depth = 4 on main train & test....")
# acc_train_train = test_tree(encode_array(train), encode_array(train), 4)
# acc_train_test = test_tree(encode_array(train), encode_array(test), 4)
# print("Results for depth = 4 on train ....")
# print(acc_train_train)
# print("Testing for depth = 4 on test....")
# print(acc_train_test)