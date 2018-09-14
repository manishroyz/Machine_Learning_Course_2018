
from data import Data
import numpy as np
from main import test_tree, encode_array

file1 = ".//data_new//CVfolds_new//fold1.csv"
file2 = ".//data_new//CVfolds_new//fold1.csv"
file3 = ".//data_new//CVfolds_new//fold1.csv"
file4 = ".//data_new//CVfolds_new//fold1.csv"
file1 = ".//data_new//CVfolds_new//fold1.csv"

depth = [1, 2, 3, 4, 5, 10, 15]

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
    data_obj2 = Data(fpath = file1).raw_data
    data_obj3 = Data(fpath = file1).raw_data
    data_obj4 = Data(fpath = file1).raw_data
    data_obj5 = Data(fpath = file1).raw_data

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

    a1 = cv(train1,test1)
    a2 = cv(train2,test2)
    a3 = cv(train3,test3)
    a4 = cv(train4,test4)
    a5 = cv(train5,test5)
    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)

    a = np.vstack((a1, a2, a3, a4, a5)).T
    a_final = np.mean(a, 1)
    return [a_final, depth]





