#Sean Reyes
#Started: 11/13/26 6:25PM
#Ended: 11/14/26 12:15PM
#A. Information gain chooses applicant_id because it allows the tree to separate the applicants into small groups. The tree will predict the applicant using ID thresholds. Always predicting 'no' achieves the test accuracy from the rejected test applicants.
#B. The split information calculated is ~7.5. The gain ratio divides info gain by split information, which penalizes features that create small groups.
#C. T2 overfits the training data without improving its ability to classify applicants.

import math
from dataclasses import dataclass, field
from data_utils import(
    accuracy,
    is_numeric_column,
    to_number
)

TARGET_COLUMN = "approved"
ID_COLUMN = "applicant_id"

def majority_class(rows, target_column = TARGET_COLUMN):
    #return the most common label
    counts = {}

    for row in rows:
        label = row[target_column]
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return None

    highest_count = max(counts.values())

    return sorted(
        label for label, count in counts.items()
        if count == highest_count
    )[0]

def entropy(rows, target_column = TARGET_COLUMN):
    #calculate entropy using log base 2
    if not rows:
        return 0.0

    counts = {}

    for row in rows:
        label = row[target_column]
        counts[label] = counts.get(label, 0) + 1

    total = len(rows)
    
    result = 0.0

    for count in counts.values():
        probability = count / total
        if probability > 0:
            result -= probability * math.log2(probability)

    return result

def split_rows_categorical(rows, column):
    #split rows into one group for each categorical value
    groups = {}

    for row in rows:
        value = row[column]
        groups.setdefault(value, []).append(row)

    return groups

def split_rows_numeric(rows, column, threshold):
    #split rows into groups <= and > the threshold
    left = []
    right = []

    for row in rows:
        if float(row[column]) <= threshold:
            left.append(row)
        else:
            right.append(row)

    return {"<=": left, ">":right}

def information_gain(rows, column, target_column = TARGET_COLUMN, threshold = None):
    #calculate info gain for either categorical or numerical split
    if not rows:
        return 0.0

    if threshold is None:
        groups = split_rows_categorical(rows, column)
    else:
        groups = split_rows_numeric(rows, column, threshold)

    if len(groups) <= 1:
        return 0.0

    parent_entropy = entropy(rows, target_column)
    weighted_child_entropy = 0.0

    for group in groups.values():
        weighted_child_entropy += (
            len(group) / len(rows)
        ) * entropy(group, target_column)

    return parent_entropy - weighted_child_entropy

def split_info(rows, column, threshold = None):
    #calculate split info
    if not rows:
        return 0.0

    if threshold is None:
        groups = split_rows_categorical(rows, column)
    else:
        groups = split_rows_numeric(rows, column, threshold)

    total = len(rows)
    result = 0.0

    for group in groups.values():
        probability = len(group) / total
        if probability > 0:
            result -= probability * math.log2(probability)

    return result

def gain_ratio(rows, column, target_column = TARGET_COLUMN, threshold = None):
    #calculate gain ratio
    gain = information_gain(
        rows, column, target_column, threshold
    )

    scattering = split_info(rows, column, threshold)

    if scattering == 0:
        return 0.0

    return gain / scattering

def candidate_threshold(rows, column):
    #return midpoints between consecutive distinct sorted values
    values = []

    for row in rows:
        value = to_number(row[column])

        if value is None:
            raise ValueError(
                f"Column '{column}' contains non-numeric values"
            )

        values.append(value)

    unique_values = sorted(set(values))

    thresholds = []

    for index in range(len(unique_values) - 1):
        midpoint = (
            unique_values[index] + unique_values[index + 1]
        ) / 2.0

        thresholds.append(midpoint)

    return thresholds


@dataclass
class TreeNode:
    #represents a single node in a tree
    prediction: str
    sample_count: int

    feature: str = None
    threshold: float = None
    is_numeric: bool = False

    children: dict = field(default_factory = dict)

class DecisionTree:
    def __init__(
        self,
        criterion = "gain",
        max_depth = None,
        min_samples_split = 2,
        target_column = TARGET_COLUMN,
        excluded_columns = None
    ):
        #initialize the decision tree
        if criterion not in ("gain", "ratio"):
            raise ValueError(
                "criterion must be 'gain' or 'ratio'"
            )

        if max_depth is not None and max_depth < 0:
            raise ValueError(
                "max_depth cannot be negative"
            )

        if min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be at least 2"
            )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.target_column = target_column
        self.excluded_columns = set(excluded_columns or [])
        self.root = None
        self.feature_columns = []


    def fit(self, rows):
        #train the decision tree
        if not rows:
            raise ValueError(
                "cannot train a decision tree with no rows"
            )

        if self.target_column not in rows[0]:
            raise ValueError(
                "Target column was not found"
            )

        self.feature_columns = [
            column
            for column in rows[0].keys()
            if column != self.target_column
            and column not in self.excluded_columns
        ]

        if not self.feature_columns:
            raise ValueError(
                "no feature columns for training"
            )

        self.root = self._build(
            rows, 
            self.feature_columns,
            depth = 0
        )

        return self

    def _best_split(self, rows, available_features):
        #find the feature and threshold that produce the best split
        best_split = None
        best_score = -1.0

        for column in available_features:
            numeric = is_numeric_column(rows, column)

            if numeric:
                thresholds = candidate_threshold(
                    rows,
                    column
                )
            else:
                thresholds = [None]

            for threshold in thresholds:
                gain = information_gain(
                    rows,
                    column,
                    self.target_column,
                    threshold
                )
                #do not create a split that improves nothing
                if gain <= 0:
                    continue

                if self.criterion == "gain":
                    score = gain
                else:
                    scattering = split_info(
                        rows,
                        column,
                        threshold
                    )

                    if scattering == 0:
                        continue

                    score = gain / scattering

                #only replace the best split when the newest one is better
                if score > best_score + 1e-12:
                    best_split = {
                        "column": column,
                        "threshold": threshold,
                        "numeric": numeric,
                        "gain": gain,
                        "score": score
                    }
                    best_score = score
        return best_split

    def _build(self, rows, available_features, depth):
        #build the decision tree
        prediction = majority_class(
            rows,
            self.target_column
        )

        node = TreeNode(
            prediction = prediction,
            sample_count = len(rows)
        )

        labels = {
            row[self.target_column]
            for row in rows
        }

        #stops when every row has the same target class
        if len(labels) == 1:
            return node

        #stop when max depth is reached
        if (
            self.max_depth is not None
            and depth >= self.max_depth
        ):
            return node

        #stop when there are not enough rows to split
        if len(rows) < self.min_samples_split:
            return node

        #stop when there are no remaining features
        if not available_features:
            return node

        split = self._best_split(
            rows,
            available_features
        )

        #stop if no useful split can be found
        if split is None:
            return node

        node.feature = split["column"]
        node.threshold = split["threshold"]
        node.is_numeric = split["numeric"]

        if node.is_numeric:
            groups = split_rows_numeric(
                rows,
                node.feature,
                node.threshold
            )

            next_features = available_features

        else:
            groups = split_rows_categorical(
                rows,
                node.feature
            )

            next_features = [
                feature
                for feature in available_features
                if feature != node.feature
            ]

        #build each child branch
        for branch_value, group in groups.items():
            node.children[branch_value] = self._build(
                group,
                next_features,
                depth + 1
            )

        return node

    def predict_one(self, row):
        #predict the target class for one applicant
        if self.root is None:
            raise ValueError(
                "the decision tree has not been trained"
            )

        node = self.root

        while node.feature is not None:
            if node.is_numeric:
                value = to_number(
                    row.get(node.feature)
                )
                if value is None:
                    return node.prediction

                if value <= node.threshold:
                    branch = "<="
                else:
                    branch = ">"
            else:
                branch = row.get(node.feature)

            if branch not in node.children:
                return node.prediction

            node = node.children[branch]

        return node.prediction

    def predict(self, rows):
        #predict the target class for multiple rows
        return [
            self.predict_one(row)
            for row in rows
        ]

    def print_tree(self):
        #print the trained decision tree
        if self.root is None:
            print("tree hasn't been trained")
            return

        self._print_node(
            self.root,
            depth = 0
        )

    def _print_node(self, node, depth):
        #print a node and its children
        indent = "  " * depth

        #leaf node
        if node.feature is None:
            print(
                f"{indent}-> {node.prediction}"
            )

            return

        #numeric decision node
        if node.is_numeric:
            print(
                f"{indent}{node.feature} <= "
                f"{node.threshold:g}:"
            )

            self._print_node(
                node.children["<="],
                depth + 1
            )

            print(
                f"{indent}{node.feature} > "
                f"{node.threshold:g}:"
            )

            self._print_node(
                node.children[">"],
                depth + 1
            )

        #categorical decision node
        else:
            for value in sorted(node.children):
                print(
                    f"{indent}{node.feature} = {value}:"
                )

                self._print_node(
                    node.children[value],
                    depth + 1
                )

def count_leaves(node):
    #count the number of leaf nodes in a trained decision tree
    if node is None:
        return 0

    if node.feature is None:
        return 1

    return sum(
        count_leaves(child)
        for child in node.children.values()
    )

def run_experiment(
    name,
    train_rows,
    test_rows,
    criterion,
    drop_id = False,
    max_depth = None,
    print_tree = False
):
    #train and evaluate one decision tree experiement
    excluded = []

    if drop_id:
        excluded.append(ID_COLUMN)

    model = DecisionTree(
        criterion = criterion,
        max_depth = max_depth,
        excluded_columns = excluded
    )

    #train the models
    model.fit(train_rows)

    #generate predictions
    train_predictions = model.predict(train_rows)
    test_predictions = model.predict(test_rows)

    #extract actual labels
    train_actual = [
        row[TARGET_COLUMN]
        for row in train_rows
    ]

    test_actual = [
        row[TARGET_COLUMN]
        for row in test_rows
    ]

    #calculate results
    train_accuracy = accuracy(
        train_actual,
        train_predictions
    )

    test_accuracy = accuracy(
        test_actual,
        test_predictions
    )

    leaves = count_leaves(model.root)

    if model.root.feature is None:
        root_feature = "(leaf)"
    else:
        root_feature = model.root.feature

    #display results
    print()
    print(name)
    print("-" * len(name))
    print(f"Criterion:         {criterion}")

    if drop_id:
        print("Features:          applicant_id dropped")
    else:
        print("Features:          all available columns")

    if max_depth is None:
        print("Maximum depth:     none")
    else:
        print(f"Maximum depth:     {max_depth}")

    print(f"Root split:        {root_feature}")
    print(
        f"Training accuracy: {train_accuracy:.1f}%"
    )
    print(
        f"Test accuracy:     {test_accuracy:.1f}%"
    )
    print(f"Leaf nodes:        {leaves}")

    #print the tree if requested
    if print_tree:
        print()
        print("Decision Tree")
        print("-------------")
        model.print_tree()

    return model

def run_self_tests():
    #run small tests to verify the decision tree calculations
    #Test 1: entropy
    sample = (
        [{"approved": "yes"} for _ in range(4)]
        + [{"approved": "no"}]
    )

    expected_entropy = 0.721928094

    actual_entropy = entropy(sample)

    assert abs(
        actual_entropy - expected_entropy
    ) < 1e-8, "Entropy test failed"

    #Test 2: numeric info gain
    values = [
        540,
        580,
        610,
        640,
        700,
        760
    ]

    labels = [
        "no",
        "no",
        "yes",
        "no",
        "yes",
        "yes"
    ]

    rows = [
        {
            "credit_score": str(value),
            "approved": label
        }
        for value, label in zip(values, labels)
    ]

    expected_gains = {
        560.0: 0.1908745046,
        595.0: 0.4591479170,
        625.0: 0.0817041659,
        670.0: 0.4591479170,
        730.0: 0.1908745046
    }

    for threshold, expected in expected_gains.items():
        actual = information_gain(
            rows,
            "credit_score",
            threshold = threshold
        )

        assert abs(
            actual - expected
        ) < 1e-8, (
            f"Info test failed at {threshold}"
        )

    #Test 3: candidate threshold
    thresholds = candidate_threshold(
        rows,
        "credit_score"
    )

    expected_thresholds = [
        560.0,
        595.0,
        625.0,
        670.0,
        730.0
    ]

    assert thresholds == expected_thresholds, (
        "Candidate threshold test failed"
    )

    #Test 4: Best numeric threshold
    model = DecisionTree(
        criterion = "gain"
    )

    best = model._best_split(
        rows,
        ["credit_score"]
    )

    assert best is not None
    assert best["threshold"] == 595.0, (
        "Best split test failed"
    )

    #Test 5: basic model training and prediction
    training_rows = [
        {"feature": "A", "approved": "yes"},
        {"feature": "A", "approved": "yes"},
        {"feature": "B", "approved": "no"},
        {"feature": "B", "approved": "no"}
    ]

    test_model = DecisionTree(
        criterion = "gain"
    )

    test_model.fit(training_rows)

    predictions = test_model.predict(training_rows)

    assert predictions == [
        "yes",
        "yes",
        "no",
        "no"
    ], "Basic prediction test failed"

    print("decision tree test passed")

def run_all_experiments(train_rows, test_rows):
    #run all 4 dedicated Decision Tree experiments
    print()
    print("=" * 55)
    print("DECISION TREE EXPERIMENTS")
    print("=" * 55)

    #T1
    run_experiment(
        name = "T1 - Information gain, all columns",
        train_rows = train_rows,
        test_rows = test_rows,
        criterion = "gain"
    )

    #T2
    run_experiment(
        name = "T2 - Information gain, ID dropped",
        train_rows = train_rows,
        test_rows = test_rows,
        criterion = "gain",
        drop_id = True
    )

    #T3
    run_experiment(
        name = "T3 - Gain ratio, all columns",
        train_rows = train_rows,
        test_rows = test_rows,
        criterion = "ratio"
    )

    #T4
    run_experiment(
        name = "T4 - Information gain, ID dropped, Depth 3",
        train_rows = train_rows,
        test_rows = test_rows,
        criterion = "gain",
        drop_id = True,
        max_depth = 3,
        print_tree = True
    )
            
        
                
                            

    
        
            

    