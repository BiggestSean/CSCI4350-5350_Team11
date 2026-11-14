#Sean Reyes
#Started: 11/13/26 5:03PM
#Ended: 11/13/26 6:24PM

import csv
import math
import os

def find_data_file(filename):
    #find the data set (.csv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Could not find {filename} in the same directory"
        )

    return file_path

def load_csv(filename):
    #reads the csv file
    file_path = find_data_file(filename)
    
    with open(filename, "r", newline = "", encoding = "utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]

def to_number(value):
    #return a float if the value is numeric
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def is_numeric_column(rows, column):
    #checks if col is numbers
    if not rows:
        return False

    for row in rows:
        if to_number(row[column]) is None:
            return False

    return True

def accuracy(actual, predicted):
    #calculate classification accuracy
    if len(actual) == 0:
        return 0.0

    correct = sum(
        actual_value == predicted_value
        for actual_value, predicted_value in zip(actual, predicted)
    )

    return (correct / len(actual)) * 100.00

def rmse(actual, predicted):
    #calculate root mean squared error
    if len(actual) == 0:
        return 0.0

    squared_error = [
        (actual_value - predicted_value) ** 2
        for actual_value, predicted_value in zip(actual, predicted)
    ]

    return math.sqrt(sum(squared_errors) / len(squared_error))

def r2(actual, predicted):
    #calculate the coeffiecients of determination
    if len(actual) == 0:
        return 0.0

    mean_actual = sum(actual) / len(actual)

    total_sum_squares = sum(
        (value - mean_actual) ** 2 for value in actual
    )

    residual_sum_squares = sum(
        (actual_value - predicted_value) ** 2
        for actual_value, predicted_value in zip(actual, predicted)
    )

    if total_sum_squares == 0:
        return 0.0

    return 1 - (residual_sum_squares / total_sum_squares)

def standardize_train_test(train_rows, test_rows, columns):
    #standardize numeric columns using training-set stats
    standardized_train = [row.copy() for row in train_rows]
    standardized_test = [row.copy() for row in test_rows]
    statistics = {}

    for column in columns:
        train_values = [float(row[column]) for row in train_rows]
        mean_value = sum(train_values) / len(train_values)
        variance = sum(
            (value - mean_value) ** 2 for value in train_values
        ) / len(train_values)

        standard_deviation = math.sqrt(variance)

        if standard_deviation == 0:
            standard_deviation = 1.0

        statistics[column] = {
            "mean": mean_value,
            "standard_deviation": standard_deviation
        }

        for row in standardized_train:
            row[column] = (
                float(row[column]) - mean_value
            ) / standard_deviation

        for row in standardized_test:
            row[column] = (
                float(row[column]) - mean_value
            ) / standard_deviation

    return standardized_train, standardized_test, statistics

def purity(cluster_assignments, true_labels):
    #calculate clustering purity
    if len(cluster_assignments) == 0:
        return 0.0

    clusters = {}

    for cluster, label in zip(cluster_assignmnets, true_labels):
        clusters.setdefault(cluster, {})
        clusters[cluster][label] = clusters[cluster].get(label, 0) + 1

    correct = sum(
        max(label_counts.values())
        for label_counts in clusters.values()
    )

    return correct / len(cluster_assignmnets)