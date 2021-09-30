#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.


    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into
    the structured DST output.

    We then run the main DST Evaluation script to get results.
"""
import argparse
import json
from ambigous.utils.convert import parse_disambiguation_label_from_file
from sklearn.metrics import recall_score, accuracy_score,f1_score, precision_score


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_target", help="path for target, line-separated format (.txt)"
    )
    parser.add_argument(
        "--input_path_predicted",
        help="path for model prediction output, line-separated format (.txt)",
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_disambiguation_label_from_file(input_path_target)
    list_predicted = parse_disambiguation_label_from_file(input_path_predicted)

    # Evaluate
    # report = evaluate_from_flat_list(list_target, list_predicted)
    # accuracy = sum([not(list_target[i] ^ list_predicted[i]) for i in range(len(list_target))])/len(list_target)
    recall = recall_score(list_target, list_predicted, average='binary')
    precision = precision_score(list_target, list_predicted,average='binary')
    accuracy = accuracy_score(list_target, list_predicted)
    f1 = f1_score(list_target, list_predicted)
    report_dict = {"target": list_target, "predicted": list_predicted, "accuracy":accuracy, "recall": recall, "f1": f1, "precision": precision}

    # Save report
    with open(output_path_report, "w") as f_out:
        json.dump(report_dict, f_out)