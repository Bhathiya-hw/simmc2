import json
import argparse

if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_graph_json", help="input path to the generated  scene graph"
    )
    parser.add_argument("--output_path_attributes", help="output path attributed")
    parser.add_argument("--output_path_predicates", help="output path predicate")
    parser.add_argument("--output_path_objects", help="output path objects")
    parser.add_argument("--output_path_object2attributes", help="output path mapping objects  to attributes")

    args = parser.parse_args()

    input_graph_json = args.input_graph_json
    output_path_attributes = args.output_path_attributes
    output_path_predicates = args.output_path_predicates
    output_path_objects = args.output_path_objects
    output_path_object2attributes = args.output_path_object2attributes
    # output_path_special = args.output_path_special

    scene_graph_load= json.load(open(input_graph_json))

    # SPECIAL_ATTRIBUTE_TOKENS = ['price = unknown', 'sleeveLength = unknown', 'color = unknown', 'type = unknown', 'size = unknown',
    #                            'brand = unknown', 'customerReview = unknown', 'pattern = unknown', 'materials = unknown', 'DISAMBIGUATE = YES', 'DISAMBIGUATE = NO']

    SPECIAL_EDGE_TOKENS = ['price', 'sleeveLength', 'color', 'type', 'materials', 'brand', 'pattern', 'positioned', 'assetType', 'availableSizes', 'customerRating', 'customerReview']
    SPECIAL_ATTRIBUTE_TOKENS  = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    # DISAMBIGUATE_NODE_TOKEN = ['DISAMBIGUATE = YES', 'DISAMBIGUATE = NO' ]
    # MENTIONED_OBJECTS_TOKENS = ['MENTIONED = YES', 'MENTIONED = NO']


    attributes = set()
    predicates = set()
    objects = set()
    # special_tokens = SPECIAL_EDGE_TOKENS + SPECIAL_ATTRIBUTE_TOKENS + DISAMBIGUATE_NODE_TOKEN
    object2attributes = {}
    for json_path, graph in scene_graph_load.items():
         for object_index, object_graph in graph.items():

             objects.add(object_index)

             visual = object_graph['visual']
             non_visual  = object_graph['non-visual']
             # prefab_path = object_graph['prefab']
             relations = object_graph['relation']

             unique_id_attr = str(object_graph['unique_id'])
             attributes.add(unique_id_attr)

             object_attr = visual + non_visual + [unique_id_attr]
             object2attributes[object_index] = object_attr

             for vis_rel in visual:
                attributes.add(vis_rel)

             for non_vis_rel in non_visual:
                attributes.add(non_vis_rel)

             # for attribute_token in SPECIAL_ATTRIBUTE_TOKENS:
             #    attributes.add(attribute_token)

             for edge_token in SPECIAL_EDGE_TOKENS:
                predicates.add(edge_token)

             for relation in relations:
                rel_predicate = relation[0]
                predicates.add(rel_predicate)


    with open(output_path_attributes, 'w+') as f:
        json.dump(SPECIAL_ATTRIBUTE_TOKENS,f)

    with open(output_path_predicates, 'w+') as f:
        json.dump(list(predicates),f)

    with open(output_path_objects, 'w+') as f:
        json.dump(list(objects),f)

    with open(output_path_object2attributes, 'w+') as f:
        json.dump(object2attributes,f)

    # with open(output_path_special, 'w+') as f:
    #     json.dump(special_tokens,f)