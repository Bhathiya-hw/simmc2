#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m glimmer_coref.scripts.graph_generate.scene_graph_generation \
--sg_output="${PATH_DIR}"/glimmer_coref/data/graph_data/scene_graph.json  \
--fashion_json_input="${PATH_DIR}"/../../data/fashion_prefab_metadata_all.json \
--furniture_json_input="${PATH_DIR}"/../../data/furniture_prefab_metadata_all.json \
