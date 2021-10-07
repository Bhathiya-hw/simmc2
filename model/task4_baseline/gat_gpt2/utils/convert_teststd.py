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

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "Belief State :"
START_OF_RESPONSE = "<SOR> Response: => "
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE} {belief_state} {END_OF_BELIEF} {START_OF_RESPONSE}"
TEMPLATE_TARGET = "{context} {START_BELIEF_STATE} {belief_state} {END_OF_BELIEF} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_OF_RESPONSE} "
TEMPLATE_TARGET_NOBELIEF = "{context} {START_OF_RESPONSE} {response} {END_OF_SENTENCE}"


def convert_json_to_flattened(
    input_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=True,
    use_belief_states=False,
    use_sys_belief_state =True,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]

    predicts = []
    targets = []
    scenes = []
    beliefs = []
    disambiguate_labels = []
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
        special_tokens["additional_special_tokens"] = additional_special_tokens

    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()

    for _, dialog in enumerate(data):

        prev_asst_uttr = None
        prev_turn = None
        lst_context = []

        for turn in dialog[FIELDNAME_DIALOG]:

            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            # user_belief = turn[FIELDNAME_BELIEF_STATE]

            sys_belief = turn[FIELDNAME_SYSTEM_STATE]

            # if 'act' not in list(sys_belief.keys()):
            #     continue
            if FIELDNAME_ASST_UTTR in list(turn.keys()):
                asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()

            # Format main input context
            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "

            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Concat with previous contexts
            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])

            relvant_turn = max([int(i) for i in dialog['scene_ids'].keys() if int(i) <= turn['turn_idx']])
            scene = dialog['scene_ids'][str(relvant_turn)]
            scenes.append(scene)

            if use_sys_belief_state and  'act' in list(sys_belief.keys()):
                belief_state = []
                # for bs_per_frame in user_belief:
                str_belief_state_per_frame = (
                    "{act} [ {slot_values} ] ({request_slots}) < {objects} >".format(
                        act=sys_belief["act"].strip(),
                        slot_values=", ".join(
                            [
                                f"{k.strip()} = {str(v).strip()}"
                                for k, v in sys_belief["act_attributes"][
                                    "slot_values"
                                ].items()
                            ]
                        ),
                        request_slots=", ".join(
                            sys_belief["act_attributes"]["request_slots"]
                        ),
                        objects=", ".join(
                            # add_visual_descriptions(scene_graph[scene + '_scene.json'],sys_belief["act_attributes"]["objects"])
                            [str(o) for o in sys_belief["act_attributes"]["objects"]]
                        ),
                    )
                )
                turn_attribute = []
                belief_state.append(str_belief_state_per_frame)
                turn_attribute.append(sys_belief['act'])
                for slot, val in sys_belief["act_attributes"]["slot_values"].items():
                    if type(val) is dict:
                        turn_attribute.append(slot)
                        for k,v in val.items():
                            if type(v) is list:
                                turn_attribute.append(','.join(v))
                            else:
                                turn_attribute.append(k + " = " + str(v))
                    else:
                        turn_attribute.append(slot + " = " + str(val))
                for rslot  in sys_belief["act_attributes"]["request_slots"]:
                    turn_attribute.append(rslot)
                for object in sys_belief["act_attributes"]["objects"]:
                    turn_attribute.append( "Object ID: " + str(object))
                beliefs.append(','.join(turn_attribute))
                # Track OOVs
                if output_path_special_tokens != "":
                    oov.add(sys_belief["act"])
                    for slot_name in sys_belief["act_attributes"]["slot_values"]:
                        oov.add(str(slot_name))
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)

                str_belief_state = " ".join(belief_state)

                # Format the main input
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                    belief_state=str_belief_state,
                    END_OF_BELIEF=END_OF_BELIEF,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
                predicts.append(predict)

                # Format the main output
                # target = TEMPLATE_TARGET.format(
                #     context=context,
                #     START_BELIEF_STATE=START_BELIEF_STATE,
                #     belief_state=str_belief_state,
                #     END_OF_BELIEF=END_OF_BELIEF,
                #     START_OF_RESPONSE=START_OF_RESPONSE,
                #     response=asst_uttr,
                #     END_OF_SENTENCE=END_OF_SENTENCE,
                # )
                # targets.append(target)

            else:
                # Format the main input
                predict = TEMPLATE_PREDICT_NOBELIEF.format(
                    context=context, START_OF_RESPONSE=START_OF_RESPONSE
                )
                # predicts.append(predict)

                # # Format the main output
                # target = TEMPLATE_TARGET_NOBELIEF.format(
                #     context=context,
                #     response=asst_uttr,
                #     END_OF_SENTENCE=END_OF_SENTENCE,
                #     START_OF_RESPONSE=START_OF_RESPONSE,
                # )
                # targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # directory = os.path.dirname(output_path_target)
    # if not os.path.exists(directory):
    #     os.makedirs(directory, exist_ok=True)

    with open(output_path_predict, "w", encoding="utf-8") as f_predict:
        X = "\n".join(predicts)
        f_predict.write(X)

    # with open(output_path_target, "w", encoding="utf-8") as f_target:
    #     Y = "\n".join(targets)
    #     f_target.write(Y)

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov))
            json.dump(special_tokens, f_special_tokens)

def add_visual_descriptions(graph, object_ids):
    feature_strings = []
    for o in object_ids:
        slot_value_pairs = {}
        feature_string = str(o) + ': '
        graph_key = 'Object ID: ' + str(o)
        if graph_key not in list(graph.keys()):
            feature_string += '{}:'.format(str(o))
            continue
        visual_features = graph['Object ID: ' + str(o)]['visual']
        for visual_feature in visual_features:
            slot_value = visual_feature.split('=')
            slot = slot_value[0].strip()
            value = slot_value[1].strip()
            slot_value_pairs[slot] = value
        non_visual_features = graph['Object ID: ' + str(o)]['non-visual']
        for non_visual_feature in non_visual_features:
            slot_value = non_visual_feature.split('=')
            slot = slot_value[0].strip()
            value = slot_value[1].strip()
            slot_value_pairs[slot] = value

        slot_keys = list(slot_value_pairs.keys())
        ordered_terms = []
        if 'pattern' in slot_keys:
            ordered_terms.append(slot_value_pairs['pattern'])
        if 'color' in slot_keys:
            ordered_terms.append(slot_value_pairs['color'])
        if 'material' in slot_keys:
            ordered_terms.append(slot_value_pairs['material'])
        ordered_terms.append(slot_value_pairs['type'])
        if 'brand' in slot_keys:
            ordered_terms.append('by ' + slot_value_pairs['brand'])
        if 'customerRating' in slot_keys:
            rating = slot_value_pairs['customerRating']
            if float(rating)>= 4.0:
                ordered_terms.append('with good ratings')
        if 'customerReview' in slot_keys:
            rating = slot_value_pairs['customerReview']
            if float(rating)>= 4.0:
                ordered_terms.append('with good reviews')
        ordered_terms.append('priced ' + slot_value_pairs['price'])
        if 'positioned' in slot_keys:
            ordered_terms.append(slot_value_pairs['positioned'])

        feature_string += ' '.join(ordered_terms)
        # if ('positioned', 'pattern') in list(slot_value_pairs.keys()):
        #     feature_string += '{}: {} {} {} {}'.format(str(o), slot_value_pairs['color'], slot_value_pairs['pattern'], slot_value_pairs['type'], slot_value_pairs['positioned'])
        # elif 'pattern' in list(slot_value_pairs.keys()):
        #     feature_string += '{}: {} {} {}'.format(str(o), slot_value_pairs['color'], slot_value_pairs['pattern'], slot_value_pairs['type'])
        # else:
        #     feature_string += '{}: {} {}'.format(str(o), slot_value_pairs['color'], slot_value_pairs['type'])
        feature_strings.append(feature_string)
    return feature_strings

def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)

    # target_attributes = ['pos', 'color', 'type', 'class_name', 'decor_style']
    #
    # list_str_objects = []
    # for obj_name, obj in visual_objects.items():
    #     s = obj_name + ' :'
    #     for target_attribute in target_attributes:
    #         if target_attribute in obj:
    #             target_value = obj.get(target_attribute)
    #             if target_value == '' or target_value == []:
    #                 pass
    #             else:
    #                 s += f' {target_attribute} {str(target_value)}'
    #     list_str_objects.append(s)
    #
    # str_objects = ' '.join(list_str_objects)

    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def parse_flattened_results_from_file(path):
    results = []
    with open(path, "r") as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

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
        r"([\w:?.?]*)  *\[([^\]]*)\] *\(([^\]]*)\) *\<([^\]]*)\>"
    )
    slot_regex = re.compile(r"([A-Za-z0-9_.-:]*)  *= ([^,]*)")
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
