import math

class DecisionTreeClassifier:
    def __init__(self):
        self.id3 = None
        self.id3_bagged = []

    def fit(self, data_train, depth_limit=3, depth=0):
        # print('Training Started')
        x = data_train[:, 1:]
        y = data_train[:, 0]

        data_train = format_data(data_train)

        features = []
        for i in range(x.shape[1]):
            feat = ':'.join([str(i), '0,1'])
            features.append(feat)
        features = feature_info(features)

        self.id3 = ID3(data_train, features, depth, depth_limit)
        # print('Training Done')

    def predict(self, data_test):
        data_test = format_data(data_test)
        preds = []
        for d in data_test:
            output = walk_down(self.id3, d[0], d[1])
            if output > 0:
                preds.append(1)
            else:
                preds.append(0)
        return preds

def format_data(data):
    X = data[:, 1:]
    y = data[:, 0]

    for i, a in enumerate(y):
        if a > 0:
            y[i] = 1
        else:
            y[i] = 0

    data_ = []
    for i, x in enumerate(X):
        data_.append([x, y[i]])
    return data_


def walk_down(node, point, label):
    if node.name == "leaf":
        return node.value
    if node.branches:
        for b in node.branches:
            if b.value == point[node.index]:
                return walk_down(b.child, point, label)
    return 0


def ID3(data_samples, attributes, depth, depth_limit):
    if not attributes or depth == depth_limit:
        leaf = Node()
        leaf.set_is_leaf(most_common(data_samples))
        return leaf

    if (all_same(data_samples)):
        label = data_samples[0][1]
        root = Node()
        root.set_is_leaf(label)
        return root

    base_entropy = calculate_base_entropy(data_samples)
    root = best_attribute(data_samples, base_entropy, attributes)
    root = Node(root.name, root.possible_vals, root.index)
    depth += 1

    for val in root.possible_vals:
        b = Branch(val)
        root.add_branch(b)
        subset = subset_val(data_samples, root.index, val)
        if not subset:
            leaf = Node()
            leaf.set_is_leaf(most_common(data_samples))
            b.set_child(leaf)
        else:
            attributes = remove_attribute(attributes, root)
            b.set_child(ID3(subset, attributes, depth, depth_limit))
    return root


def best_attribute(data, base_entropy, attributes):
    max_ig = 0
    max_a = None
    for a in attributes:
        tmp_ig = base_entropy - expected_entropy(data, a)
        tmp_a = a
        if tmp_ig >= max_ig:
            max_ig = tmp_ig
            max_a = a
    return max_a


# Returns the most common label
def most_common(data_samples):
    p = sum(d[1] for d in data_samples)
    if p >= len(data_samples) / 2:
        return 1
    else:
        return 0


def expected_entropy(data, attribute):
    data_total = float(len(data))
    e_entropy = 0.0
    for val in attribute.possible_vals:
        entropy, total = calculate_entropy(data, attribute, val)
        e_entropy += (total / data_total) * entropy
    return e_entropy


def calculate_entropy(data, attribute, value):
    subset = subset_val(data, attribute.index, value)
    if not subset:
        return [0, 0]

    return [calculate_base_entropy(subset), len(subset)]


def calculate_base_entropy(data):
    l = len(data)
    p = sum(d[1] for d in data)

    if not p or l == p:
        return 0

    n = l - p

    probP = p / l
    probN = n / l

    return (-probP * math.log(probP)) - (probN * math.log(probN))


# Returns a subset of the data where the given feature has the given value
def subset_val(data, feature_index, val):
    return [d for d in data if d[0][feature_index] == val]


# Returns true if all the labels are the same in the sample data
def all_same(data_samples):
    label = data_samples[0][1]
    for s in data_samples:
        if s[1] != label:
            return False
    return True


def remove_attribute(attributes, attribute):
    new_attributes = []
    for a in attributes:
        if a.name != attribute.name:
            new_attributes.append(a)
    return new_attributes


def feature_info(data):
    data_inf = []
    for i, d in enumerate(data):
        d = d.split(":")
        r = list(map(int, d[1].rstrip().split(",")))
        a = Node(d[0], r, i)
        data_inf.append(a)

    return data_inf


class Node:
    def __init__(self, name="leaf", vals=None, index=-1):
        self.name = name
        self.possible_vals = vals
        self.index = index
        self.branches = []

    def set_is_leaf(self, value):
        self.leaf = True
        self.value = value

    def add_branch(self, b):
        self.branches.append(b)


class Branch:
    def __init__(self, value):
        self.value = value

    def set_child(self, child):
        self.child = child
