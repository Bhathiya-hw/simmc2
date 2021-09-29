#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    echo $PATH_DIR
else
    PATH_DIR=$(realpath "$1")
    echo $PATH_DIR
fi

--output_dir
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/save/task23/taskDC_mock
--model_type
gpt2
--model_name_or_path
gpt2
--line_by_line
--add_special_tokens
/home/bhash/GAT2GPT/simmc2/model//task_2/gat_coref/data/dst_coref/simmc2_special_tokens.json




# Train (multi-modal)
python3 -m gat_coref.scripts.run_language_modeling \
  --output_dir="${PATH_DIR}"/gat_coref/save/dc_g4  \
  --model_type=gpt2 \
  --model_name_or_path=gpt2 \
  --line_by_line \
  --add_special_tokens="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_special_tokens.json \
  --do_train  \
  --train_data_file="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_dials_dstc10_dev_target.txt \
  --predict_train_data_file="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_dials_dstc10_dev_predict.txt \
  --scene_train_data_file="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_dials_dstc10_dev_scene.txt \
  --graph_json_file="${PATH_DIR}"/gat_coref/data/dst_coref/graph_data/scene_graph.json \
  --do_eval \
  --eval_data_file="${PATH_DIR}"/gat_coref/data/dst_coref/simmc2_dials_dstc10_dev_target.txt  \
  --num_train_epochs=2  \
  --overwrite_output_dir  \
  --per_gpu_train_batch_size=1  \
  --per_gpu_eval_batch_size=1 \
  --save_steps=5000 \
  --gat_conv_layers=4\



