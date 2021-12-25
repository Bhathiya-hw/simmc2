import json
import pathlib


ROOT_DIR = pathlib.Path('/home/bhash/GAT2GPT/simmc2/model/task2_glimmer/glimmer_coref/data/graph_data')

with open(ROOT_DIR /  'objects_112.json') as f:
    OBJECTS_INV = json.load(f)
    OBJECTS = {k: i for i, k in enumerate(OBJECTS_INV)}

with open(ROOT_DIR / 'predicates_112.json') as f:
    RELATIONS_INV = json.load(f)
    RELATIONS = {k: i for i, k in enumerate(RELATIONS_INV)}

with open(ROOT_DIR / 'attributes_112.json') as f:
    ATTRIBUTES_INV = json.load(f)
    ATTRIBUTES = {k: i for i, k in enumerate(ATTRIBUTES_INV)}

with open(ROOT_DIR / 'object2attributes_112.json') as f:
    mapping = json.load(f)
