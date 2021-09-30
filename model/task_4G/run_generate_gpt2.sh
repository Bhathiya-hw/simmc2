#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m generation.scripts.run_generation \
--model_type=graph2dial  \
--model_name_or_path="${PATH_DIR}"/generation/save/with_gat/checkpoint-60 \
--num_return_sequences=1  \
--length=100  \
--stop_token='<EOS>'  \
--prompts_from_file="${PATH_DIR}"/generation/data/simmc2_dials_dstc10_devtest_predict.txt  \
--path_output="${PATH_DIR}"/generation/results/simmc2_dials_dstc10_devtest_predicted.txt \
--graph_json_file="${PATH_DIR}"/generation/data/graph_data/scene_graph.json \
--scene_file="${PATH_DIR}"/generation/data/simmc2_dials_dstc10_devtest_scene.txt \
--belief_file="${PATH_DIR}"/generation/data/simmc2_dials_dstc10_devtest_sys_belief.txt \
--gat_conv_layers=5 \
--no_cuda

