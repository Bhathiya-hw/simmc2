#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ./../../data/)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# TestStd split
python3 -m ambigous.scripts.preprocess_input_teststd \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_teststd_public.json \
    --output_path_sys_belief="${PATH_DIR}"/ambigous/data/teststd/simmc2_dials_dstc10_teststd_sys_belief.txt \
    --output_path_scene="${PATH_DIR}"/ambigous/data/teststd/simmc2_dials_dstc10_teststd_scene.txt \
    --output_path_predict="${PATH_DIR}"/ambigous/data/teststd/simmc2_dials_dstc10_teststd_predict.txt \
    --output_path_target="${PATH_DIR}"/ambigous/data/teststd/simmc2_dials_dstc10_teststd_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/ambigous/data/teststd/simmc2_special_tokens.json \
    --output_path_dt="${PATH_DIR}"/ambigous/data/teststd/simmc2_dials_dstc10_teststd_dt.txt \

