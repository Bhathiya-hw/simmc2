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
from glimmer_coref.utils.convert_disambiguate import parse_disambiguation_label_from_file
from sklearn.metrics import recall_score, accuracy_score,f1_score, precision_score


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_dt", help="path for target, line-separated format (.txt)"
    )
    parser.add_argument(
        "--input_path_predicted",
        help="path for model prediction output, line-separated format (.txt)",
    )

    parser.add_argument(
        "--output_path_results", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_dt = args.input_path_dt
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_results

    with open(input_path_dt,  encoding="utf-8") as dt:
        dts = [line for line in dt.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(input_path_predicted,  encoding="utf-8") as pred:
        preds = [line.split('=>')[1].split() for line in pred.read().splitlines() if (len(line) > 0 and not line.isspace())]

    results = []

    prev_dialogue = ''
    predictions = []
    for idx, dt in enumerate(dts):
        pred = preds[idx]
        pred_bool = 'YES' in pred
        d,t = dt.split('_')
        d = int(d)
        t = int(t)
        if prev_dialogue != d:
            if prev_dialogue != '':
                results.append({"dialog_id":int(prev_dialogue), "predictions":predictions})
                predictions = []

        predictions.append({'turn_id': t, "disambiguation_label": pred_bool})
        if idx == len(dts)-1:
            results.append({"dialog_id": int(d), "predictions": predictions})
        prev_dialogue = int(d)

    # Save report
    with open(output_path_report, "w") as f_out:
        json.dump(results, f_out)