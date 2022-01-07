#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi
# Evaluate (multi-modal)
python -m glimmer_coref.scripts.evaluate_disambiguation \
    --input_path_target="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_devtest_target.txt  \
    --input_path_predicted="${PATH_DIR}"/glimmer_coref/results/disambiguate/glimm_disambiguate.txt \
    --output_path_report="${PATH_DIR}"/glimmer_coref/results/reports/glimm_disambiguate_report.json

#python -m glimmer_coref.scripts.evaluate_response \
#    --input_path_target="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_devtest_target.txt \
#    --input_path_predicted="${PATH_DIR}"/glimmer_coref/results/simmc2_dials_dstc10_devtest_predicted.txt
