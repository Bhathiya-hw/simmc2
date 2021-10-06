#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    echo $PATH_DIR
else
    PATH_DIR=$(realpath "$1")
    echo $PATH_DIR
fi

# Train (multi-modal)
python3 -m ambigous.scripts.run_language_modeling \
  --output_dir="${PATH_DIR}"/ambigous/save/task1_baseline  \
  --model_type=gpt2\
  --model_name_or_path=gpt2\
  --line_by_line \
  --add_special_tokens="${PATH_DIR}"/ambigous/data/simmc2_special_tokens.json \
  --do_train  \
  --train_data_file="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_train_target.txt \
  --do_eval \
  --eval_data_file="${PATH_DIR}"/ambigous/data/simmc2_dials_dstc10_devtest_target.txt  \
  --num_train_epochs=5  \
  --overwrite_output_dir  \
  --per_gpu_train_batch_size=1  \
  --per_gpu_eval_batch_size=1 \
  --save_steps=10 \
  --no_cuda
