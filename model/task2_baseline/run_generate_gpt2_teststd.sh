#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m gat_coref.scripts.run_generation \
--model_type=graph2dial  \
--model_name_or_path="${PATH_DIR}"/gat_coref/save/baseline_pre-trained/checkpoint-40 \
--num_return_sequences=1  \
--length=100  \
--stop_token='<EOS>'  \
--prompts_from_file="${PATH_DIR}"/gat_coref/data/teststd/simmc2_dials_dstc10_teststd_predict.txt  \
--path_output="${PATH_DIR}"/gat_coref/results/simmc2_dials_dstc10_teststd_predicted_baseline_pre-trained.txt \
