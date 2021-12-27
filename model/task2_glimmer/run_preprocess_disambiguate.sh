#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi

# Train split
python3 -m glimmer_coref.scripts.preprocess_input_inv \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_train.json \
    --output_path_predict="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_train_predict.txt \
    --output_path_target="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_train_target.txt \
    --len_context=2 \
    --task_id=1 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/glimmer_coref/data/simmc2_special_tokens.json \
    --output_path_scene="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_train_scene.txt \
    --input_path_sg="${PATH_DIR}"/glimmer_coref/data/graph_data/scene_graph.json \
    --input_path_fashion_meta="${PATH_DATA_DIR}"/fashion_prefab_metadata_all.json \
    --input_path_furniture_meta="${PATH_DATA_DIR}"/furniture_prefab_metadata_all.json \
    --output_path_err="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_train_err.csv

# Dev split
python3 -m glimmer_coref.scripts.preprocess_input_inv \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_dev.json \
    --output_path_predict="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_dev_predict.txt \
    --output_path_target="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_dev_target.txt \
    --len_context=2 \
    --task_id=1 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/glimmer_coref/data/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/glimmer_coref/data/simmc2_special_tokens.json \
    --output_path_scene="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_dev_scene.txt \
    --input_path_sg="${PATH_DIR}"/glimmer_coref/data/graph_data/scene_graph.json \
    --input_path_fashion_meta="${PATH_DATA_DIR}"/fashion_prefab_metadata_all.json \
    --input_path_furniture_meta="${PATH_DATA_DIR}"/furniture_prefab_metadata_all.json \
    --output_path_err="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_dev_err.csv

# Devtest split
python3 -m glimmer_coref.scripts.preprocess_input_inv \
    --input_path_json="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
    --output_path_predict="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_devtest_predict.txt \
    --output_path_target="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_devtest_target.txt \
    --len_context=2 \
    --task_id=1 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/glimmer_coref/data/simmc2_special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/glimmer_coref/data/simmc2_special_tokens.json \
    --output_path_scene="${PATH_DIR}"/glimmer_coref/data/simmc2_dials_dstc10_devtest_scene.txt \
    --input_path_sg="${PATH_DIR}"/glimmer_coref/data/graph_data/scene_graph.json \
    --input_path_fashion_meta="${PATH_DATA_DIR}"/fashion_prefab_metadata_all.json \
    --input_path_furniture_meta="${PATH_DATA_DIR}"/furniture_prefab_metadata_all.json \
    --output_path_err="${PATH_DIR}"/glimmer_coref/data/disambiguate/simmc2_dials_dstc10_devtest_err.csv