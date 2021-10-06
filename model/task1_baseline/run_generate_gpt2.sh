#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m ambigous.scripts.run_generation \
--model_type=graph2dial  \
--model_name_or_path="${PATH_DIR}"/ambigous/save/task1_baseline/checkpoint-10 \
--num_return_sequences=1  \
--length=100  \
--stop_token='<EOS>'  \
--prompts_from_file="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_predict.txt  \
--path_output="${PATH_DIR}"/ambigous/results/simmc2_dials_dstc10_devtest_predicted_baseline.txt \
