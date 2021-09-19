#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m gat_coref.scripts.graph_generate.task2_scene_graph_generation \
--sg_output="${PATH_DIR}"/gat_coref/data/graph_data/t2_scene_graph.json  \
--fashion_json_input="${PATH_DIR}"/../../data/fashion_prefab_metadata_all.json \
--furniture_json_input="${PATH_DIR}"/../../data/furniture_prefab_metadata_all.json \
