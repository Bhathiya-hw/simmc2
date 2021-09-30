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
