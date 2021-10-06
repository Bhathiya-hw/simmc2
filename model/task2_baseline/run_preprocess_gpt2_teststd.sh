#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Devtest split
python3 -m gat_coref.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_teststd_public.json \
    --output_path_predict="${PATH_DIR}"/gat_coref/data/teststd/simmc2_dials_dstc10_teststd_predict.txt \
    --output_path_target="${PATH_DIR}"/gat_coref/data/teststd/simmc2_dials_dstc10_teststd_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gat_coref/data/teststd/simmc2_special_tokens.json \
