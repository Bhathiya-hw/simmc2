#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m gat_coref.scripts.task23_run_generation \
--model_type=graph2dial  \
--model_name_or_path="${PATH_DIR}"/gat_coref/save/task23/task23_mock/checkpoint-20 \
--num_return_sequences=1  \
--length=100  \
--stop_token='<EOS>'  \
--prompts_from_file="${PATH_DIR}"/gat_coref/data/t2_simmc2_dials_dstc10_devtest_target.txt  \
--path_output="${PATH_DIR}"/gat_gpt2/results/t4_simmc2_dials_dstc10_devtest_predicted.txt \
--graph_json_file="${PATH_DIR}"/gat_gpt2/data/graph_data/scene_graph.json \
--scene_file="${PATH_DIR}"/gat_gpt2/data/t4_simmc2_dials_dstc10_devtest_scene.txt \
--belief_file="${PATH_DIR}"/gat_gpt2/data/t4_simmc2_dials_dstc10_devtest_sys_belief.txt \
--gat_conv_layers=5 \
--no_cuda


--train_data_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/t2_simmc2_dials_dstc10_dev_target.txt
--predict_train_data_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/t2_simmc2_dials_dstc10_dev_predict.txt
--scene_train_data_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/t2_simmc2_dials_dstc10_dev_scene.txt
--belief_train_data_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/t2_simmc2_dials_dstc10_dev_sys_belief.txt
--graph_json_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/graph_data/t2_scene_graph.json
--do_eval
--eval_data_file
/home/bhash/GAT2GPT/simmc2/model/task_2/gat_coref/data/t1_simmc2_dials_dstc10_dev_target.txt
--num_train_epochs
5
--overwrite_output_dir
--per_gpu_train_batch_size
1
--per_gpu_eval_batch_size
1
--save_steps
10
--gat_conv_layers
2
--cache_dir
/home/bhash/.cache/huggingface/transformers
--no_cuda