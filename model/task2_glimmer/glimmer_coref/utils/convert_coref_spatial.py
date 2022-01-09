#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import re
import os
import glimmer_coref.scripts.graph_representation.Constants as Constants
import csv

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_OF_CATALOG_CONTEXTS = "<SCAT>"
END_OF_CATALOG_CONTEXTS = "<ECAT>"
START_OF_SPATIAL_CONTEXTS = "<SREL>"
END_OF_SPATIAL_CONTEXTS = "<EREL>"
START_OF_PRED_CATALOG = "<SPCT>"
END_OF_PRED_CATALOG = "<EPCT>"
START_BELIEF_STATE = "=> Belief State :"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} "
TEMPLATE_TARGET = (
    "{context} {START_BELIEF_STATE} {belief_state} "
    "{END_OF_BELIEF} {response} {END_OF_SENTENCE}"
)

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=True,
    input_path_retrieval=None,
    output_path_retrieval=None,
    input_path_special_tokens="",
    output_path_special_tokens="",
    input_path_scene_graph = "",
    output_path_scene = "",
    output_path_err = "",
    input_path_fahsion_meta = "",
    input_path_furniture_meta = ""
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    # If input_path_retrieval is not None, also encode retrieval options.
    if input_path_retrieval is not None:
        with open(input_path_retrieval, "r") as file_id:
            retrieval_options = json.load(file_id)
        format_retrieval_options = True
        options_pool = retrieval_options["system_transcript_pool"]
        options_dict = {
            ii["dialogue_idx"]: ii
            for ii in retrieval_options["retrieval_candidates"]
        }
        retrieval_targets = []
    else:
        format_retrieval_options = False

    append_unique_ids = False
    if input_path_scene_graph is not None and output_path_scene is not None:
        with open(input_path_scene_graph, 'r') as file_id:
            scene_graph =  json.load(file_id)
            print(scene_graph.keys())
            append_unique_ids = True

    with open(input_path_fahsion_meta, 'r') as fas:
        fashion_meta = json.load(fas)

    with open(input_path_furniture_meta, 'r') as fur:
        furniture_meta = json.load(fur)

    prefabs2ind = {pf:"INV_"+str(i) for i,pf in enumerate(list(fashion_meta.keys()) +list(furniture_meta.keys()))}
    ind2prefab = {v:k for k,v in prefabs2ind.items()}

    predicts = []
    targets = []
    valid_coref = []
    additional_special_tokens = []
    if input_path_special_tokens != "":
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {"eos_token": END_OF_SENTENCE}
        if use_belief_states:
            additional_special_tokens.append(END_OF_BELIEF)
        else:
            additional_special_tokens.append(START_OF_RESPONSE)
        if use_multimodal_contexts:
            additional_special_tokens.extend(
                [START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS]
            )
    additional_special_tokens.extend(Constants.OBJECTS_INV)
    additional_special_tokens.extend(Constants.ATTRIBUTES_INV)
    additional_special_tokens.extend(list(ind2prefab.keys()))
    special_tokens["additional_special_tokens"] = additional_special_tokens

    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()
    scene_list = []
    err_list = []
    for _, dialog in enumerate(data):

        domain = dialog["domain"]
        dialog_id = dialog["dialogue_idx"]
        prev_asst_uttr = None
        prev_turn = None
        lst_context = []

        for turn_id, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            err_dict = dict()
            err_dict['domain'] = domain
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

            if append_unique_ids:
                relvant_turn = max([int(i) for i in dialog['scene_ids'].keys() if int(i) <= turn['turn_idx']])
                scene = dialog['scene_ids'][str(relvant_turn)]
                if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                    scene_list.append(scene + "_scene.json")
            # Format main input context
            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "
                if use_multimodal_contexts:
                    # Add multimodal contexts
                    visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                        "act_attributes"
                    ]["objects"]
                    if append_unique_ids:
                        visual_object_rep = represent_visual_objects(visual_objects, scene_graph[scene + "_scene.json"], prefabs2ind, ind2prefab)
                        context += visual_object_rep
                    else:
                        context += represent_visual_objects(visual_objects, None)

            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Add multimodal contexts -- user shouldn't have access to ground-truth
            """
            if use_multimodal_contexts:
                visual_objects = turn[FIELDNAME_BELIEF_STATE]['act_attributes']['objects']
                context += ' ' + represent_visual_objects(visual_objects)
            """

            # Concat with previous contexts
            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])
            err_dict['context'] = context
            # Format belief state
            if use_belief_states:
                belief_state = []
                # for bs_per_frame in user_belief:
                if append_unique_ids:
                    invval = [prefabs2ind[scene_graph[scene + '_scene.json']["O" + str(o)]['prefab']] for o in user_belief["act_attributes"]["objects"] if "O" + str(o) in list(scene_graph[scene + '_scene.json'].keys())]
                    for i in range(2):
                        invkey = "unique_" + str(i)
                        if i < len(invval):
                            err_dict[invkey] = invval[i]
                        else:
                            err_dict[invkey] = None
                    objval = ["O" + str(o) for o in user_belief["act_attributes"]["objects"]]
                    for i in range(2):
                        objkey = "object_" + str(i)

                        if i < len(objval):
                            err_dict[objkey] = objval[i]
                            if objval[i] in scene_graph[scene + '_scene.json'].keys():
                                print(objval[i])
                                err_dict['visual_' + str(i) ] =scene_graph[scene + '_scene.json'][str(objval[i])]['visual']
                                err_dict['non-visual_' + str(i)] = scene_graph[scene + '_scene.json'][str(objval[i])]['non-visual']
                                err_dict['relation_'+ str(i)] = scene_graph[scene + '_scene.json'][str(objval[i])]['relation']
                            else:
                                err_dict['visual_'+ str(i)] = "Unknown"
                                err_dict['non-visual_' + str(i)] = "Unknown"
                                err_dict['relation_' + str(i)] = "Unknown"

                        else:
                            err_dict[objkey] = None
                            err_dict['visual_' + str(i)] = None
                            err_dict['non-visual_' + str(i)] = None
                            err_dict['relation_' + str(i)] = None
                    str_belief_state_per_frame = (
                        "{act} [ {slot_values} ] ({request_slots}) < {objects} > {uniques} ".format(
                            act=user_belief["act"].strip(),
                            slot_values=", ".join(
                                [
                                    f"{k.strip()} = {str(v).strip()}"
                                    for k, v in user_belief["act_attributes"][
                                        "slot_values"
                                    ].items()
                                ]
                            ),
                            request_slots=", ".join(
                                user_belief["act_attributes"]["request_slots"]
                            ),

                            uniques=START_OF_PRED_CATALOG + " " + ", ".join(
                                [prefabs2ind[scene_graph[scene + '_scene.json']["O" + str(o)]['prefab']] for o in user_belief["act_attributes"]["objects"] if "O" + str(o) in list(scene_graph[scene + '_scene.json'].keys())]
                            ) + " " + END_OF_PRED_CATALOG,

                            objects=", ".join(
                                ["O" + str(o) for o in user_belief["act_attributes"]["objects"]]
                            )
                        )
                    )
                else:
                    str_belief_state_per_frame = (
                        "{act} [ {slot_values} ] ({request_slots}) < {objects} >".format(
                            act=user_belief["act"].strip(),
                            slot_values=", ".join(
                                [
                                    f"{k.strip()} = {str(v).strip()}"
                                    for k, v in user_belief["act_attributes"][
                                        "slot_values"
                                    ].items()
                                ]
                            ),
                            request_slots=", ".join(
                                user_belief["act_attributes"]["request_slots"]
                            ),
                            objects=", ".join(
                                [str(o) for o in user_belief["act_attributes"]["objects"]]
                            ),
                        )
                    )
                belief_state.append(str_belief_state_per_frame)

                # Track OOVs
                if output_path_special_tokens != "":
                    oov.add(user_belief["act"])
                    for slot_name in user_belief["act_attributes"]["slot_values"]:
                        oov.add(str(slot_name))
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)

                str_belief_state = " ".join(belief_state)

                # Format the main input
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                )
                if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                    predicts.append(predict)
                    valid_coref.append(True)
                else:
                    valid_coref.append(False)
                # Format the main output
                target = TEMPLATE_TARGET.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                    belief_state=str_belief_state,
                    END_OF_BELIEF=END_OF_BELIEF,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                )
                if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                    targets.append(target)

                # NOTE: Retrieval options w/ belief states is not implemented.
            else:
                # Format the main input
                predict = TEMPLATE_PREDICT_NOBELIEF.format(
                    context=context, START_OF_RESPONSE=START_OF_RESPONSE
                )
                if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                    predicts.append(predict)
                    valid_coref.append(True)
                else:
                    valid_coref.append(False)
                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    context=context,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
                if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                    targets.append(target)

                # Add retrieval options is necessary.
                if format_retrieval_options:
                    turn_options = (
                        options_dict[dialog_id]["retrieval_candidates"][turn_id]
                    )
                    for option_ind in turn_options["retrieval_candidates"]:
                        retrieval_target = TEMPLATE_TARGET_NOBELIEF.format(
                            context=context,
                            response=options_pool[domain][option_ind],
                            END_OF_SENTENCE=END_OF_SENTENCE,
                            START_OF_RESPONSE=START_OF_RESPONSE,
                        )
                        retrieval_targets.append(retrieval_target)
            if turn['system_transcript_annotated']['act'] != 'REQUEST:DISAMBIGUATE':
                err_list.append(err_dict)
    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    keys = err_list[0].keys()
    with open(output_path_err, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter = "~")
        dict_writer.writeheader()
        dict_writer.writerows(err_list)

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        X = "\n".join(predicts)
        f_predict.write(X)

    with open(output_path_target, "w") as f_target:
        Y = "\n".join(targets)
        f_target.write(Y)

    if append_unique_ids:
        with open(output_path_scene, "w") as f_scene:
            S = "\n".join(scene_list)
            f_scene.write(S)
    output_path_valid =  output_path_target.replace('_target.txt', '_valid.json')
    with open(output_path_valid, "w") as f_valid:
        json.dump(valid_coref,f_valid)
        # V = "\n".join(valid_coref)
        # f_valid.write(V)
    # Write retrieval candidates if necessary.
    if format_retrieval_options:
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_retrieval)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(output_path_retrieval, "w") as file_id:
            file_id.write("\n".join(retrieval_targets))

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


def represent_visual_objects(object_ids, scene=None, prefab2ind=None, ind2Prefab=None):
    # Stringify visual objects (JSON)
    """
    target_attributes = ['pos', 'color', 'type', 'class_name', 'decor_style']

    list_str_objects = []
    for obj_name, obj in visual_objects.items():
        s = obj_name + ' :'
        for target_attribute in target_attributes:
            if target_attribute in obj:
                target_value = obj.get(target_attribute)
                if target_value == '' or target_value == []:
                    pass
                else:
                    s += f' {target_attribute} {str(target_value)}'
        list_str_objects.append(s)

    str_objects = ' '.join(list_str_objects)
    """
    if scene is not None:
        str_objects = ", ".join(["O"+str(o) for o in object_ids])
        unique_ids = ', '.join([prefab2ind[scene["O"+str(o)]['prefab']] for o in object_ids if "O"+str(o) in list(scene.keys())])
        all_object = [k for k,v in scene.items() if v['inventory_id'] in [scene["O"+ str(obj_id)]['inventory_id'] for  obj_id in object_ids if "O"+str(obj_id) in list(scene.keys())] ]
        obj_relations = [(o,scene[o]['relation']) for o in all_object]
        triplet_dict = dict()
        triplet_list = list()
        for triplet in obj_relations:
            subj, relations = triplet
            related_relations = [r  for r in relations if r[1] in all_object]
            for rr in related_relations:
                triplet_dict[subj] = (subj, rr[0], rr[1])
                triplet_list.append((subj, rr[0], rr[1]))
        print(triplet_dict)
        triplet_strings = ', '.join([f'{r[0]} {r[1].replace(" " , "")} {r[2]}' for r in triplet_list])

        return f"{START_OF_CATALOG_CONTEXTS} {unique_ids} {END_OF_CATALOG_CONTEXTS}  {START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS} {START_OF_SPATIAL_CONTEXTS} {triplet_strings} {END_OF_SPATIAL_CONTEXTS} "
        # str_objects = ", ".join(["O" + str(o) for o in object_ids])
        # unique_ids = ', '.join([scene["O" + str(o)]['unique_id'] for o in object_ids if "O" + str(o) in list(scene.keys())])
        # return f"{START_OF_CATALOG_CONTEXTS} {unique_ids} {END_OF_CATALOG_CONTEXTS} "
    else:
        str_objects = ", ".join([str(o) for o in object_ids])
        return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"

def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results

def parse_disambiguation_label_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            label = line.split('=>')[1].replace('<EOS>', '').strip() == 'YES' #parse_flattened_result(line)
            results.append(label)

    return results


def parse_flattened_result(to_parse):
    """
    Parse out the belief state from the raw text.
    Return an empty list if the belief state can't be parsed

    Input:
    - A single <str> of flattened result
      e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

    Output:
    - Parsed result in a JSON format, where the format is:
        [
            {
                'act': <str>  # e.g. 'DA:REQUEST',
                'slots': [
                    <str> slot_name,
                    <str> slot_value
                ]
            }, ...  # End of a frame
        ]  # End of a dialog
    """
    dialog_act_regex = re.compile(
        r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>'
    )    
    
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= (\[(.*)\]|[^,]*)")
    request_regex = re.compile(r"([A-Za-z0-9_.-:]+)")
    object_regex = re.compile(r"([A-Za-z0-9]+)")

    belief = []

    # Parse
    splits = to_parse.strip().split(START_BELIEF_STATE)
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split(END_OF_BELIEF)

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    "act": dialog_act.group(1),
                    "slots": [],
                    "request_slots": [],
                    "objects": [],
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d["slots"].append([slot.group(1).strip(), slot.group(2).strip()])

                for request_slot in request_regex.finditer(dialog_act.group(3)):
                    d["request_slots"].append(request_slot.group(1).strip())

                for object_id in object_regex.finditer(dialog_act.group(4)):
                    d["objects"].append(object_id.group(1).strip())

                if d != {}:
                    belief.append(d)

    return belief


if __name__ == '__main__':
    print('-')
    to_parse = "=> Belief State : INFORM:GET [ sleeveLength = short, availableSizes = ['XXL', 'S', 'L'], pattern = leafy design ] (availableSizes, pattern) < 86, 57 > <EOB>"
    print(to_parse)
    print(parse_flattened_result(to_parse))

    print('-')
    to_parse = "=> Belief State : INFORM:GET [ sleeveLength = short, availableSizes = ['XXL', 'S', 'L'] ] (availableSizes) < 86, 57 > <EOB>"
    print(to_parse)
    print(parse_flattened_result(to_parse))

    print('-')
    to_parse = "=> Belief State : INFORM:GET [ sleeveLength = short, pattern = leafy ] (availableSizes) < 86, 57 > <EOB>"
    print(to_parse)
    print(parse_flattened_result(to_parse))
