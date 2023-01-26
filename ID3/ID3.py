import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts, unique_vals
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        total_counts = sum(counts.values())
        probabilities = [num/total_counts for num in counts.values()]
        impurity = - sum(prob * np.log2(prob) for prob in probabilities)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        #Note in this implementation, the split only creates 2 children at a time

        total_examples = len(left) + len(right)
        entropy_L = self.entropy(left, left_labels)
        entropy_R = self.entropy(right, right_labels)
        info_gain_value = current_uncertainty - (len(left)/total_examples) * entropy_L - (len(right)/total_examples) * entropy_R
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        true_rows, true_labels, false_rows, false_labels = [],[],[],[]
        for index, example in enumerate(rows):
            if question.match(example):
                true_rows.append(example)
                true_labels.append(labels[index])
            else:
                false_rows.append(example)
                false_labels.append(labels[index])
        
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================set(self)

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        #We use dynamic auto-discretization to find the best data split at any given time
        # available_features = set(self.label_names) - {self.target_attribute} - self.used_features
        features = self.label_names[1:]

        for feature_index, feature in enumerate(features):
            monotonic_vals = list(unique_vals(rows, feature_index))
            monotonic_vals.sort()
            k = len(monotonic_vals)
            thresholds = [0.5 * (monotonic_vals[i] + monotonic_vals[i+1]) for i in range(k - 1)]

            for threshold in thresholds:
                question = Question(feature, feature_index, threshold)

                gain, true_rows, true_labels, false_rows, false_labels \
                    = self.partition(rows, labels, question, current_uncertainty)

                if gain >= best_gain: #need to have '=' here to ensure max index is reached as set out by the homework
                    best_gain, best_question, best_false_rows, best_false_labels, best_true_rows, best_true_labels =\
                        gain, question, false_rows, false_labels, true_rows, true_labels
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        # First we need to check if the current node should not be split again:
        available_features = set(self.label_names) - {self.target_attribute} - self.used_features
        if (len(rows) < self.min_for_pruning) or (self.entropy(rows, labels) == 0) or (len(available_features) == 0):
            return Leaf(rows, labels)

        #recursive step 
        _, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = self.find_best_split(rows, labels)
        # self.used_features.add(best_question.column)
        true_branch = self.build_tree(best_true_rows, best_true_labels)
        false_branch = self.build_tree(best_false_rows, best_false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        if type(node) == Leaf:
            return max(node.predictions, key=node.predictions.get)
        
        else:
            predicted_match = node.question.match(row)
            selected_branch = node.true_branch if predicted_match else node.false_branch
            prediction = self.predict_sample(row, selected_branch)
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.array([self.predict_sample(row, self.tree_root) for row in rows])
        # ========================

        return y_pred
