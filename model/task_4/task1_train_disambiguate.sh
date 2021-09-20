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
python3 -m gat_gpt2.scripts.task4_run_language_modeling \
  --output_dir="${PATH_DIR}"/gat_gpt2/save/task1_gat4_b2  \
  --model_type=gpt2\
  --model_name_or_path=gpt2\
  --line_by_line \
  --add_special_tokens="${PATH_DIR}"/gat_gpt2/data/task1_data/simmc2_special_tokens.json \
  --do_train  \
  --train_data_file="${PATH_DIR}"/gat_gpt2/data/task1_data/t1_simmc2_dials_dstc10_train_target.txt \
  --predict_train_data_file="${PATH_DIR}"/gat_gpt2/data/task1_data/t1_simmc2_dials_dstc10_train_predict.txt \
  --scene_train_data_file="${PATH_DIR}"/gat_gpt2/data/task1_data/t1_simmc2_dials_dstc10_train_scene.txt  \
  --belief_train_data_file="${PATH_DIR}"/gat_gpt2/data/task1_data/t1_simmc2_dials_dstc10_train_sys_belief.txt  \
  --graph_json_file="${PATH_DIR}"/gat_gpt2/data/graph_data/new_scene_graph.json \
  --do_eval \
  --eval_data_file="${PATH_DIR}"/gat_gpt2/data/task1_data/t1_simmc2_dials_dstc10_devtest_target.txt  \
  --num_train_epochs=5  \
  --overwrite_output_dir  \
  --per_gpu_train_batch_size=2  \
  --per_gpu_eval_batch_size=1 \
  --save_steps=2500 \
  --gat_conv_layers=4\
