#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
python -m gat_gpt2.scripts.graph_generate.graph_encode_tokens \
--input_graph_json="${PATH_DIR}"/gat_gpt2/data/graph_data/scene_graph.json  \
--output_path_attributes="${PATH_DIR}"/gat_gpt2/data/graph_data/attributes.json \
--output_path_predicates="${PATH_DIR}"/gat_gpt2/data/graph_data/predicates.json \
--output_path_objects="${PATH_DIR}"/gat_gpt2/data/graph_data/objects.json \
--output_path_object2attributes=/"${PATH_DIR}"/gat_gpt2/data/graph_data/object2attributes.json \
