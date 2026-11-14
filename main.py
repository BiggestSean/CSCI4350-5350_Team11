#Sean Reyes
#Started: 11/14/26 12:17PM
#Ended: 11/14/26 2:49PM
#Decision tree stuffs only

import argparse
import os

from data_utils import load_csv
from decision_tree import (
    run_experiment,
    run_all_experiments,
    run_self_tests
)

def main():
    parser = argparse.ArgumentParser(
        description = "Programming Assignment 3"
    )

    parser.add_argument(
        "--algo",
        choices = ["tree"]
    )

    parser.add_argument(
        "--criterion",
        choices = ["gain", "ratio"]
    )

    parser.add_argument(
        "--drop-id",
        action = "store_true"
    )

    parser.add_argument(
        "--max-depth",
        type = int,
        default = None
    )

    parser.add_argument(
        "--print-tree",
        action = "store_true"
    )

    parser.add_argument(
        "--experiments",
        action = "store_true"
    )

    parser.add_argument(
        "--skip-self-tests",
        action = "store_true"
    )

    args = parser.parse_args()

    #run small example tests first
    if not args.skip_self_tests:
        run_self_tests()

    #load both datasets
    train_rows = load_csv("loan_train.csv")
    test_rows = load_csv("loan_test.csv")

    print("\nDATASET INFORMATION")
    print("-------------------")
    print(f"Training Rows: {len(train_rows)}")
    print(f"Testing Rows: {len(test_rows)}")

    #run the decision tree tests
    if args.experiments:
        run_all_experiments(train_rows, test_rows)
    else:
        run_experiment(
            name = "Decision Tree",
            train_rows = train_rows,
            test_rows = test_rows,
            criterion = args.criterion,
            drop_id = args.drop_id,
            max_depth = args.max_depth,
            print_tree = args.print_tree
        )

if __name__ == "__main__":
    main()

