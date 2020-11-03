"""
Original source code is from pyfpgrowth library, but added our own method here which is get_classification_association_rule and rankRules.
This new addition generates CARs and rank the rules using our information-content formula. The information content formula is:

    * Information-content = 1/ [ log_2(Average Information Gain) + log_2(Average correlation) ]

Authors: Team What On Earth
        (Cheryl Neoh Yi Ming, Vhera Kaey Vijayaraj & Low Yi Xian)

Supervised by: Dr Ong Huey Fang
"""

# Inbuilt imports used by the source code pyfpgrowth
import itertools, math

# Inbuilt imports used by our new methods, functions
from datapreprocess import correlation
import pandas as pd

"""

FP-GROWTH SOURCE CODE

"""

class FPNode(object):
    """
    A node in the FP tree.
    """

    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions, threshold, root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(
            transactions, root_value,
            root_count, self.frequent, self.headers)

    @staticmethod
    def find_frequent_items(transactions, threshold):
        """
        Create a dictionary of items with occurrences above the threshold.
        """
        items = {}

        for transaction in transactions:
            for item in transaction:
                if item in items:
                    items[item] += 1
                else:
                    items[item] = 1

        for key in list(items.keys()):
            if items[key] < threshold:
                del items[key]

        return items

    @staticmethod
    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value,
                     root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)

    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])

    def mine_patterns(self, threshold):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            return self.zip_patterns(self.mine_sub_trees(threshold))

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if
        we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                new_patterns[tuple(sorted(list(key) + [suffix]))
                             ] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """
        patterns = {}
        items = self.frequent.keys()

        # If we are in a conditional tree,
        # the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = [self.root.value]
            patterns[tuple(suffix_value)] = self.root.count

        for i in range(1, len(items) + 1):
            for subset in itertools.combinations(items, i):
                pattern = tuple(sorted(list(subset) + suffix_value))
                patterns[pattern] = \
                    min([self.frequent[x] for x in subset])

        return patterns

    def mine_sub_trees(self, threshold):
        """
        Generate subtrees and mine them for patterns.
        """
        patterns = {}
        mining_order = sorted(self.frequent.keys(),
                              key=lambda x: self.frequent[x])

        # Get items in tree in reverse order of occurrences.
        for item in mining_order:
            suffixes = []
            conditional_tree_input = []
            node = self.headers[item]

            # Follow node links to get a list of
            # all occurrences of a certain item.
            while node is not None:
                suffixes.append(node)
                node = node.link

            # For each occurrence of the item,
            # trace the path back to the root node.
            for suffix in suffixes:
                frequency = suffix.count
                path = []
                parent = suffix.parent

                while parent.parent is not None:
                    path.append(parent.value)
                    parent = parent.parent

                for i in range(frequency):
                    conditional_tree_input.append(path)

            # Now we have the input for a subtree,
            # so construct it and grab the patterns.
            subtree = FPTree(conditional_tree_input, threshold,
                             item, self.frequent[item])
            subtree_patterns = subtree.mine_patterns(threshold)

            # Insert subtree patterns into main patterns dictionary.
            for pattern in subtree_patterns.keys():
                if pattern in patterns:
                    patterns[pattern] += subtree_patterns[pattern]
                else:
                    patterns[pattern] = subtree_patterns[pattern]

        return patterns


def find_frequent_patterns(transactions, support_threshold):
    """
    Given a set of transactions, find the patterns in it
    over the specified support threshold.
    """
    # the input into find_frequent_patterns :
    #   a) transactions :- training data
    #   b) support threshold :- number of items in your transactions that satisfies the frequent itemset
    tree = FPTree(transactions, support_threshold, None, None)
    return tree.mine_patterns(support_threshold)


def generate_association_rules(patterns, confidence_threshold):
    """
    Given a set of frequent itemsets, return a dict
    of association rules in the form
    {(left): ((right), confidence)}
    """
    # the input into generate_association_rules :
    #   a) patterns :- from above
    #   b) confidence_threshold :- the minimum threshold
    rules = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold:
                        rules[antecedent] = (consequent, confidence)

    return rules


"""

NEW METHODS CREATED

"""


def get_classification_association_rules(training_data, class_label, support_threshold, confidence_threshold):
    """ This function generates Classification Association Rules (CARs) using the fp-growth source code.

    Args:
        training_data (list): A list of transactions
        class_label (list): A list of the class labels of our target in the transaction
        support_threshold (int): The minimum support threshold for generating the frequent patterns
        confidence_threshold (int): The minimum confidence threshold for the CARs

    Returns:
        dict: A dictionary containing all the Classification Association Rules (CARs)
    """

    patterns = find_frequent_patterns(training_data, support_threshold)

    CARs = {}
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                # obtaining the rules where only the consequent has the class label
                if len(consequent) == 1 and consequent[0] in class_label:
                    if antecedent in patterns:
                        lower_support = patterns[antecedent]
                        confidence = float(upper_support) / lower_support
                        support = upper_support/len(training_data)

                        # filtering the rules where the confidence of the rules does not satisfy the minimum threshold
                        if confidence >= confidence_threshold:
                            CARs[antecedent] = (consequent, confidence, support)

    return CARs


def rankRule(data, CARs, info_gain, class_column, feature_info, use_conf = False, use_supp = False, use_conf_supp = False):
    """
    This function is used for rule ranking using the information-content formula.

    Args:
        data (list): A list of transactions
        CARs (dict): A dictionary containing all the Classification Association Rules (CARs)
        info_gain (dict): A dictionary where the key is the index of the column and the value is the information gain.
        class_column (int): the index of the target column
        feature_info (dict): A dictionary containing all the unique values each attribute has
        use_conf (bool, optional): If True then it will use confidence of the rules for ranking the rules. Defaults to False.
        use_supp (bool, optional): If True then it will use support of the rules for ranking the rules. Defaults to False.
        use_conf_supp (bool, optional): If True then it will use confidence and then support of the rules for ranking the rules. Defaults to False.

    Returns:
        [type]: [description]
    """
    ranked_CARs = []
    corr = correlation(data, class_column)

    for rule in CARs:
        consequent, confidence, support = CARs.get(rule)

        #  Get the attributes of the rule
        key_list = []
        for feature in rule:
            for key in feature_info.keys():
               if feature in feature_info[key]:
                   key_list.append(key)

        #  Calculate the average information gain and correlation of the current rule
        total_ig = 0
        total_corr = 0
        for i in range(len(key_list)):
            for item in info_gain:
                if key_list[i] == item:
                    # for information gain
                    total_ig += info_gain[item]
                    # for correlation
                    total_corr += corr[key_list[i]]
        average_ig = total_ig/len(key_list)
        average_corr = total_corr/len(key_list)

        # Calculate information-content using the average ig and correlation calculated earlier
        if average_corr < 0:
            interestingess = math.log2(1/average_ig) - math.log2(1/-average_corr)
        else:
            interestingess = math.log2(1/average_ig) + math.log2(1/average_corr)

        # Prepare the dataset for rule ranking of the rules based on the user's preference
        if use_conf:
            ranked_CARs.append( ([interestingess, confidence], rule, consequent, confidence, support, interestingess) )
        elif use_supp:
            ranked_CARs.append( ([interestingess, support], rule, consequent, confidence, support, interestingess) )
        elif use_conf_supp:
            ranked_CARs.append( ([interestingess, confidence, support], rule, consequent, confidence, support, interestingess) )
        else:
            ranked_CARs.append( ([interestingess], rule, consequent, confidence, support, interestingess) )

    # Rule ranking
    ranked_CARs = sorted(ranked_CARs, key = lambda item: (item[0]), reverse=True)
    ranked_CARs_dict = {}
    for _, antecedent, consequent, confidence, support, interestingness in ranked_CARs:
        ranked_CARs_dict[antecedent] = (consequent, confidence, support, interestingness)


    return ranked_CARs_dict
