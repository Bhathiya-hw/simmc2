#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

python3 -m  ambigous.scripts.evaluation.evaluate \
	--input_path_target="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_target.txt \
	--input_path_predicted="${PATH_DIR}"/ambigous/results/simmc2_dials_dstc10_devtest_predicted_gat.txt \
	--output_path_report="${PATH_DIR}"/ambigous/results/gat__results.json
