#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ./../../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

python3 -m  ambigous.scripts.evaluation.evaluate \
	--input_path_target=/home/hsb2000/workspace/simmc2/model/task_1/ambigous/data/simmc2_dials_dstc10_devtest_target.txt \
	--input_path_predicted=/home/hsb2000/workspace/simmc2/model/task_1/ambigous/results/simmc2_dials_dstc10_devtest_predicted_gat.txt \
	--output_path_report=/home/hsb2000/workspace/simmc2/model/task_1/ambigous/results/gat__results.json
