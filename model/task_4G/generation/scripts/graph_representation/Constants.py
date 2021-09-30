import json
import pathlib


ROOT_DIR = pathlib.Path('/home/bhash/GAT2GPT/simmc2/model/task_4/gat_gpt2/data/graph_data')

with open(ROOT_DIR /  'objects.json') as f:
    OBJECTS_INV = json.load(f)
    OBJECTS = {k: i for i, k in enumerate(OBJECTS_INV)}

with open(ROOT_DIR / 'predicates.json') as f:
    RELATIONS_INV = json.load(f)
    RELATIONS = {k: i for i, k in enumerate(RELATIONS_INV)}

with open(ROOT_DIR / 'attributes.json') as f:
    ATTRIBUTES_INV = json.load(f)
    ATTRIBUTES = {k: i for i, k in enumerate(ATTRIBUTES_INV)}

with open(ROOT_DIR / 'object2attributes.json') as f:
    mapping = json.load(f)
