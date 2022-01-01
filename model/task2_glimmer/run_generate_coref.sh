#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi
# Generate sentences (Furniture, multi-modal)
python -m glimmer_coref.scripts.run_generation \
--model_type=graph2dial  \
--model_name_or_path="${PATH_DIR}"/glimmer_coref/save/glimm_coref/checkpoint-50 \
--num_return_sequences=1  \
--length=100  \
--stop_token='<EOS>'  \
--prompts_from_file="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_devtest_predict.txt  \
--path_output="${PATH_DIR}"/gat_coref/results/both_inv_predicted_cp135.txt \
--graph_json_file="${PATH_DIR}"/glimmer_coref/data/graph_data/scene_graph.json \
--scene_file="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_devtest_scene.txt \
--gat_conv_layers=4 \
--special_encode_uniques
