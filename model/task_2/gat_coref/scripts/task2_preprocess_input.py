#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Scripts for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import gat_coref.utils.task2_convert as task2
import argparse

if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path_json", help="input path to the original dialog data"
    )
    parser.add_argument("--output_path_target", help="output path for full target")
    parser.add_argument("--output_path_predict", help="output path for prediction")
    parser.add_argument(
        "--input_path_special_tokens",
        help="input path for special tokens. blank if not provided",
        default="",
    )
    parser.add_argument(
        "--output_path_special_tokens",
        help="output path for special tokens. blank if not saving",
        default="",
    )
    parser.add_argument(
        "--len_context",
        help="# of turns to include as dialog context",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--use_multimodal_contexts",
        help="determine whether to use the multimodal contexts each turn",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_target = args.output_path_target
    output_path_predict = args.output_path_predict
    input_path_special_tokens = args.input_path_special_tokens
    output_path_special_tokens = args.output_path_special_tokens
    len_context = args.len_context
    use_multimodal_contexts = bool(args.use_multimodal_contexts)

    # Convert the data into flatten dictionary friendly format

    task2.convert_json_to_flattened(
        input_path_json,
        output_path_target,
        output_path_predict,
        input_path_special_tokens=input_path_special_tokens,
        output_path_special_tokens=output_path_special_tokens,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
    )
