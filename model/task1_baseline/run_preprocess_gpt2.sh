#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ./../../data/)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Train split
python3 -m ambigous.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_train.json \
    --output_path_predict="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_train_predict.txt \
    --output_path_target="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_train_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
    --output_path_dt="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_train_dt.txt \

# Dev split
python3 -m ambigous.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_dev.json \
    --output_path_predict="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_dev_predict.txt \
    --output_path_target="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_dev_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
    --output_path_dt="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_dev_dt.txt \
# Devtest split
python3 -m ambigous.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
    --output_path_predict="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_predict.txt \
    --output_path_target="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
    --output_path_dt="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_dt.txt \
