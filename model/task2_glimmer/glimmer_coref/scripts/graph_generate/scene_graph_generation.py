import json,copy
from os import listdir
from os.path import isfile, join

import argparse

if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()

    parser.add_argument("--fashion_json_input", help="fashion meta output")
    parser.add_argument("--furniture_json_input", help="fashion meta input")
    parser.add_argument("--sg_output", help="output path for generated scene graph")

    args = parser.parse_args()

    fashion_json_input = args.fashion_json_input
    furniture_json_input = args.furniture_json_input
    sg_output = args.sg_output

#@TODO take as arg
    fashion_meta = json.load(open(fashion_json_input))
    furniture_meta = json.load(open(furniture_json_input))

    object_graphs = {}
    for key in fashion_meta:
        object_node = fashion_meta[key]
        #fashion_visual
        fashion_non_vis = []
        fashion_non_vis.extend([ 'domain = fashion', 'customerReview = ' +  str(object_node['customerReview']),
                                  'brand = ' + object_node['brand'],
                                  'price = ' + str(object_node['price']),
                                  'size = ' + object_node['size'],
                               'sleeveLength = ' + object_node['sleeveLength']])
        fashion_non_vis.append(('availableSizes = ' + ",".join(object_node['availableSizes'])))
        #fashion visual
        fashion_vis = []
        asset_Type = object_node['assetType'].split('_')
        fashion_vis.extend(['pattern = ' + object_node['pattern'],
                              'type = ' + object_node['type'],
                             'assetType =' + object_node['assetType'],
                              'category = '+ asset_Type[0],
                              'color = ' + object_node['color']])
        if len(asset_Type)>1:
            fashion_vis.append('positioned = '+ asset_Type[1])

        # object_graphs[key] = []
        object_graphs[key] ={ 'non-visual': fashion_non_vis, 'visual': fashion_vis}
        # object_graphs[key].append({('fashion', 'visual'): fashion_vis})

    for key in furniture_meta:
        object_node = furniture_meta[key]
        #furniture_non_visual
        furniture_non_vis = []
        furniture_non_vis.extend([ 'domain = furniture','customerRating = ' + str(object_node['customerRating']),
                                  'brand = ' + object_node['brand'],
                                  'price = ' + object_node['price'],
                                  'materials = ' + object_node['materials']])
        #furniture visual
        furniture_vis = []
        furniture_vis.extend([
                              'type = ' + object_node['type'],
                              'color =' + object_node['color']])

        # object_graphs[key] = []
        object_graphs[key] = {'non-visual': furniture_non_vis, 'visual': furniture_vis}
        # object_graphs[key].append({('furniture', 'visual'): furniture_vis})

    scene_graph = {}
    image_paths = ['/home/hsb2000/workspace/glimmer/simmc2/data/public/']



    for path in image_paths:
        scene_path = [f for f in listdir(path) if isfile(join(path, f)) and ('scene.json' in f)]

        for scene in scene_path:
            # ROOT_OBJECT = {'non-visual': ['customerReview = unknown', 'brand = unknown', 'price = unknown', 'size = unknown', 'sleeveLength = unknown', 'availableSizes = unknown', 'material_unknown'],
            #                                    'visual': ['pattern = unknown', 'type = unknown', 'assetType = unknown', 'category = unknown', 'color = unknown', 'positioned = unknown']}
            # ROOT_OBJECT['unique_id'] = -1
            # ROOT_OBJECT['relation'] = []

            scene_json = path + scene
            scene_data = json.load(open(scene_json))
            scene_graph[scene] = {}
            if len(scene_data['scenes'])>1:
                print("More scenes.............")
            object_relationships = {}
            rel_map = {"down": "is under", "up": "is over", "left": "left of", "right": "right of"}
            relationships = scene_data['scenes'][0]['relationships']
            for relation in relationships:
                for tail_obj, head_objs in relationships[relation].items():
                    for head_obj in head_objs:
                        head_node_id = "O" + str(head_obj)
                        tail_node_id = "O" + str(tail_obj)
                        if head_node_id in object_relationships.keys():
                            object_relationships[head_node_id].append((rel_map[relation], tail_node_id))
                        else:
                            object_relationships[head_node_id] = [(rel_map[relation], tail_node_id)]

            for instance in scene_data['scenes'][0]['objects']:
                # scene_graph[scene]['ORT'] = ROOT_OBJECT.copy()
                if instance['prefab_path'] in object_graphs.keys():
                    # print(object_graphs[instance['prefab_path']])
                    instance_node_id = "O" + str(instance['index'])
                    scene_graph[scene][instance_node_id] = object_graphs[instance['prefab_path']].copy()
                    scene_graph[scene][instance_node_id]['prefab'] = instance['prefab_path']
                    scene_graph[scene][instance_node_id]['unique_id'] = "U" + str(instance['unique_id'])
                    scene_graph[scene][instance_node_id]['bbox'] = instance['bbox']
                    if 'relation' in scene_graph[scene][instance_node_id].keys():
                      print("Not empty")
                    if instance_node_id in object_relationships.keys():
                        scene_graph[scene][instance_node_id]['relation'] = object_relationships[instance_node_id]
                    else:
                        scene_graph[scene][instance_node_id]['relation'] = []



    # print(scene_graph)
    with open(sg_output, 'w') as f:
      json.dump(scene_graph, f)
