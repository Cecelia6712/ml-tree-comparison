import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ID3DecisionTree:
    """支持连续特征的ID3树（信息增益 + 二分法）"""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth and depth >= self.max_depth) or n_samples < self.min_samples_split or n_labels == 1:
            return Node(value=self._most_common_label(y))

        best_feat, best_thresh, best_gain = self._best_split(X, y)
        if best_gain == 0:
            return Node(value=self._most_common_label(y))

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feat, best_thresh = None, None
        parent_entropy = self._entropy(y)
        n_samples = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                e_left = self._entropy(y[left_idx])
                e_right = self._entropy(y[right_idx])
                child_entropy = (n_left/n_samples)*e_left + (n_right/n_samples)*e_right
                gain = parent_entropy - child_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist[hist>0] / len(y)
        return -np.sum(ps * np.log2(ps))

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

class CARTDecisionTree:
    """CART 分类树（基尼系数）"""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _gini(self, y):
        hist = np.bincount(y)
        probs = hist[hist>0] / len(y)
        return 1 - np.sum(probs**2)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feat, best_thresh = None, None
        n_samples = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx)==0 or np.sum(right_idx)==0:
                    continue
                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[right_idx])
                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                gini_split = (n_left/n_samples)*gini_left + (n_right/n_samples)*gini_right
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh, best_gini

    def _grow_tree(self, X, y, depth):
        n_samples, n_feats = X.shape
        if (self.max_depth and depth>=self.max_depth) or n_samples<self.min_samples_split or len(np.unique(y))==1:
            return Node(value=Counter(y).most_common(1)[0][0])
        feat, thresh, _ = self._best_split(X, y)
        if feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)