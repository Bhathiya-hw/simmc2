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

    scene_graph_load= json.load(open(input_graph_json))

    attributes = set()
    predicates = set()
    objects = set()
    object2attributes = {}
    for json_path, graph in scene_graph_load.items():
         for object_index, object_graph in graph.items():

             objects.add(object_index)

             visual = object_graph['visual']
             non_visual  = object_graph['non-visual']
             prefab_path = object_graph['prefab']
             relations = object_graph['relation']

             object_attr = visual + non_visual
             object2attributes[object_index] = object_attr

             for vis_rel in visual:
                attributes.add(vis_rel)

             for non_vis_rel in non_visual:
                attributes.add(non_vis_rel)

             for relation in relations:
                rel_predicate = relation[0]
                predicates.add(rel_predicate)

    with open(output_path_attributes, 'w+') as f:
        json.dump(list(attributes),f)

    with open(output_path_predicates, 'w+') as f:
        json.dump(list(predicates),f)

    with open(output_path_objects, 'w+') as f:
        json.dump(list(objects),f)

    with open(output_path_object2attributes, 'w+') as f:
        json.dump(object2attributes,f)