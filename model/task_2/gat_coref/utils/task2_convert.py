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

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"
FIELDNAME_ACT_ATTRIBUTE = 'act_attributes'
FIELDNAME_OBJECTS = 'objects'

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "Belief State :"
START_OF_RESPONSE = "Response: => "
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

START_OF_SLOT_VALUES = "<SSV>"
END_OF_SLOT_VALUES = "<ESV>"
START_OF_REQUESTED_SLOTS = "<SRS>"
END_OF_REQUESTED_SLOTS ="<ERS>"
START_OF_COREF = "<SOCR>"
END_OF_COREF = "<EOCR>"

START_OBJECTS_LIST = "<SOBL>"
END_OBJECTS_LIST = "<EOBL>"
START_GRAPH_INPUT = "<SGI>"
END_GRAPH_INPUT = "<EGI>"



def convert_json_to_flattened(
    input_path_json,
    output_path_target,
    output_path_predict,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=False,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    if input_path_special_tokens != "":
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {"eos_token": END_OF_SENTENCE}
        additional_special_tokens = []
        if use_belief_states:
            additional_special_tokens.append(END_OF_BELIEF)
        else:
            additional_special_tokens.append(START_OF_RESPONSE)
        if use_multimodal_contexts:
            additional_special_tokens.extend(
                [START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS]
            )
        additional_special_tokens.extend(
            [START_OF_SLOT_VALUES, END_OF_SLOT_VALUES,START_OF_REQUESTED_SLOTS, END_OF_REQUESTED_SLOTS, START_OF_COREF, END_OF_COREF,
             START_OBJECTS_LIST,  END_OBJECTS_LIST, START_GRAPH_INPUT, END_GRAPH_INPUT]
        )
        special_tokens["additional_special_tokens"] = additional_special_tokens


    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()
    target_dicts = {}
    predict_dict = {}
    predict_keys = ['user_utterance','prev_asst_utterance','visual_objects','scene', 'turn_idx' , 'dialog_idx']
    for _, dialog in enumerate(data):
        dialog_json = {}
        prev_asst_uttr = None
        prev_turn = None
        for turn in dialog[FIELDNAME_DIALOG]:
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            # if not user_belief["act_attributes"]["objects"]:
            #     user_belief["act_attributes"]["objects"] = ['RT']
            dialogue_dict_key= str(dialog['dialogue_idx']) + "_" + str(turn['turn_idx'])
            turn_dict = {}
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()


            # asst_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            turn_dict['user_utterance'] = f"User : {user_uttr}"
            # turn_dict['asst_utterance'] = f"System : {asst_uttr}"
            turn_dict['current_act'] =  turn[FIELDNAME_BELIEF_STATE]['act'].strip()
            turn_dict['slot_values'] = " ".join([START_OF_SLOT_VALUES] +
                                        [

                                            f"{k.strip()} = {str(v).strip()}"
                                            for k, v in user_belief["act_attributes"][
                                                "slot_values"
                                            ].items()

                                        ]
                                        + [END_OF_SLOT_VALUES]
                                    )
            turn_dict['requested_slots'] = " ".join( [START_OF_REQUESTED_SLOTS] +
                                        user_belief["act_attributes"]["request_slots"] + [END_OF_REQUESTED_SLOTS]
                                    )
            turn_dict['ref_object_count'] = len(user_belief["act_attributes"]["objects"])
            turn_dict['ref_objects'] = " ".join( [START_OF_COREF] +
                ["O"+ str(o) for o in user_belief["act_attributes"]["objects"]] + [END_OF_COREF]
            )

            # turn_dict[asst_uttr] = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

            if prev_asst_uttr:
                turn_dict['prev_asst_utterance'] =  f"System : {prev_asst_uttr} "
                if use_multimodal_contexts:
                    # Add multimodal contexts
                    visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                        "act_attributes"
                    ]["objects"]
                    if len(visual_objects)>0:
                        turn_dict['visual_objects'] = represent_visual_objects(visual_objects)

            prev_asst_uttr = asst_uttr
            prev_turn = turn
            current_turn_idx = turn['turn_idx']
            bottom_turn_idx = max(current_turn_idx-len_context,0)
            for prior_turn_idx in range(current_turn_idx-1,bottom_turn_idx-1,-1):
                prior_turn_key = 'prior_turn_' + str(prior_turn_idx)
                prior_flatten_dict_key = str(dialog['dialogue_idx']) + "_" + str(prior_turn_idx) #"turn_" + str(prior_turn_idx)
                turn_dict[prior_turn_key] = prior_flatten_dict_key #dialog_json[prior_turn_key]

            relvant_turn = max([int(i) for i in dialog['scene_ids'].keys() if int(i)<=turn['turn_idx']])
            scene = dialog['scene_ids'][str(relvant_turn)]
            turn_dict['scene'] = scene
            turn_dict['turn_idx'] = turn['turn_idx']
            turn_dict['dialog_idx'] = dialog['dialogue_idx']

            current_turn_key = "turn_" + str(current_turn_idx)
            dialog_json[current_turn_key] = turn_dict
            target_dicts[dialogue_dict_key]  = turn_dict

            predict_dict[dialogue_dict_key] = {k:v for k,v in turn_dict.items() if k in predict_keys}
            # Track OOVs
            if output_path_special_tokens != "":
                oov.add(user_belief["act"])
                for slot_name in user_belief["act_attributes"]["slot_values"]:
                    oov.add(str(slot_name))
                    # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                    # oov.add(slot_name)
                    # oov.add(slot_value)


    # Create a directory if it does not exist

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(output_path_target, "w", encoding="utf-8") as f_target:
            print(len(target_dicts))
            json.dump(target_dicts, f_target)

    with open(output_path_predict, "w", encoding="utf-8") as f_target:
            print(len(predict_dict))
            json.dump(predict_dict, f_target)

    # with open(output_path_target, "r", encoding="utf-8") as f_read:
    #         # print(len(flatten_dicts))
    #         # json.dump(flatten_dicts,f_target)
    #         flat_d = json.load(f_read)
    #         print(flat_d)

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)


def represent_visual_objects(object_ids):
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
    str_objects = " ".join(["O"+ str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"
