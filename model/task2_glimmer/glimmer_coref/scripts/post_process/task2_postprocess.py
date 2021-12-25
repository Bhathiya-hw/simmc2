import json
import argparse
import re
import heapq
if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_predicted", help="input path to the generated  scene graph"
    )

    parser.add_argument(
        "--input_baseline", help="input path to the generated  baseline"
    )

    parser.add_argument(
        "--input_prompt_dict", help="input path to the generated  scene graph"
    )

    parser.add_argument(
        "--input_sg", help="input path to the generated  scene graph"
    )

    parser.add_argument(
        "--output_response_path", help="input path to the generated  scene graph"
    )

    parser.add_argument(
        "--input_path_fashion_meta", help="input path to the original dialog data"
    )

    parser.add_argument(
        "--input_path_furniture_meta", help="input path to the original dialog data"
    )

    args = parser.parse_args()
    input_predicted = args.input_predicted
    prompt_dict = args.input_prompt_dict
    out_response = args.output_response_path
    input_baseline = args.input_baseline
    input_path_furniture_meta = args.input_path_furniture_meta
    input_path_fashion_meta = args. input_path_fashion_meta

    sg_path = args.input_sg

    with open(input_path_fashion_meta, 'r') as fas:
        fashion_meta = json.load(fas)

    with open(input_path_furniture_meta, 'r') as fur:
        furniture_meta = json.load(fur)

    prefabs2ind = {pf:"INV_"+str(i) for i,pf in enumerate(list(fashion_meta.keys()) +list(furniture_meta.keys()))}
    ind2prefab = {v:k for k,v in prefabs2ind.items()}


    def predict_coref(U_predicts, O_predicts, U2O_map,salency_map):
        O_final = []
        upredicts = U_predicts#U_predicts.split(' ')
        opredicts = O_predicts#O_predicts.split(' ')
        # uvisited = []
        for upredict in upredicts:
            if upredict in list(U2O_map.keys()):
                salency = salency_map[upredict]
                salency_order = heapq.nlargest(len(U2O_map[upredict]), range(len(salency)), key=salency.__getitem__)
                o_probable = [U2O_map[upredict][i] for i in salency_order]
                o_intersect = [o for o in o_probable if o in opredicts]
                if o_intersect:
                    print(len(o_intersect))
                    if o_intersect[0] in O_final and len(o_intersect)>1:
                        O_final.append(o_intersect[1])
                    else:
                        O_final.append(o_intersect[0])
                else:
                    if o_probable[0] in O_final and len(o_probable)>1:
                        O_final.append(o_probable[1])
                    else:
                        O_final.append(o_probable[0])
                    # O_final.append(o_probable[0])
            else:
                if opredicts[0] in O_final and len(opredicts)>1:
                    O_final.append(O_predicts[1])
        return O_final

    def format_sv_string(slot_val):
        # print(slot_val)
        final_str = '['
        slot_val_list = slot_val.replace('  ', ' ').replace(' = ', '=') \
            .replace('type', '##type').replace('pattern', "##pattern") \
            .replace('availableSizes', "##availableSizes").replace('size', '##size') \
            .replace('price', '##price') \
            .replace('materials', '##materials').replace('color ', '##color ') \
            .replace('customerReview', '##customerReview').replace('brand', '##brand') \
            .replace('sleeveLength', '##sleeveLength').replace('customerRating', '##customerRating').split('##')
        for pair in slot_val_list:
            if pair != '' and pair != ',' :#and 'availableSizes' not in pair:
                # print(pair)
                slot =  pair.split('=')[0].strip()
                value = pair.split('=')[1].strip()
                final_str +=  " " + slot + ' = ' + value + ','
        if ',' in final_str:
            final_str = final_str[:-1] + ' ]'
        else:
            final_str = final_str + ']'
        # print(final_str)
        return final_str
    def format_req_string(req_slot):
        # print(req_slot)
        final_str = '('
        req_slots_list = req_slot.replace('  ', ' ').replace(' = ', '=') \
            .replace('type', '##type').replace('pattern', "##pattern") \
            .replace('availableSizes', "##availableSizes").replace('size', '##size') \
            .replace('price', '##price') \
            .replace('materials', '##materials').replace('color ', '##color ') \
            .replace('customerReview', '##customerReview').replace('brand', '##brand') \
            .replace('sleeveLength', '##sleeveLength').replace('customerRating', '##customerRating').split('##')
        final_str += ', '.join([req_slot for req_slot in req_slots_list if req_slot != ''])
        final_str += ')'
        return final_str
    sg = json.load(open(sg_path))
    # prompt_dict = json.load(open(prompt_dict))

    with open(prompt_dict) as sf:
        scene_lines = sf.readlines()

    with open(input_baseline) as baseline:
        baselines = baseline.readlines()


    prefab2unique = {scene_key:{k: prefabs2ind[v['prefab']] for k, v in sg_temp.items()} for scene_key, sg_temp in sg.items() }

    prefab2obj = {}
    prefab2salency =  {}
    for scene_key, sg_temp in sg.items():
        prefab2obj_scene = {}
        prefab2salency_scene = {}
        for k, v in sg_temp.items():
            if prefabs2ind[v['prefab']] not in list(prefab2obj_scene.keys()):
                prefab2obj_scene[prefabs2ind[v['prefab']]] = [k]
                prefab2salency_scene[prefabs2ind[v['prefab']]] = [v['bbox'][2] * v['bbox'][3]]
            else:
                prefab2obj_scene[prefabs2ind[v['prefab']]].extend([k])
                prefab2salency_scene[prefabs2ind[v['prefab']]] .extend([v['bbox'][2] * v['bbox'][3]])
        prefab2obj[scene_key] = prefab2obj_scene
        prefab2salency[scene_key] = prefab2salency_scene
    with open(input_predicted, 'r') as f:
        lines = f.read().splitlines()

    responses = {}
    corefs_dict = {}
    flatten_response =  []
    #"=> Belief State : INFORM:GET [ sleeveLength = short, availableSizes = ['XXL', 'S', 'L'], pattern = leafy design ] (availableSizes, pattern) < 86, 57 > <EOB>"
    for idx, line in enumerate(lines):
        # turn_response = {}
        string_response = ''
        baseline_predicted = baselines[idx]
        if '<EOB>' not in line or '<SPCT>' not in line or '<EPCT>' not in line:
            print(idx)
            flatten_response.append(baseline_predicted[:baseline_predicted.find('<EOB>') + len('<EOB>')])
            continue

        # d_turn, content = line.split('<EOB>')[0].split('<=>')
        line = line.split('<EOB>')[0]
        u_values =  line[line.find('<SPCT>') + len('<SPCT>'):line.rfind('<EPCT>')].strip().split(' ')
        o_segment = line.split('<SPCT>')[0]
        o_values = o_segment[o_segment.rfind('<')+1:o_segment.rfind('>')].strip().split(' ')
        # print(idx, u_values)
        # print(idx, o_values)
        # dt_key = dialogue_idx + '_' + turn
        scene_key = scene_lines[idx].strip()
        coref_objects = predict_coref(u_values, o_values, prefab2obj[scene_key], prefab2salency[scene_key])
        coref_objects = list(dict.fromkeys(coref_objects))
        print(coref_objects)

        string_response += line[:line.rfind(')') +2]#baseline_predicted[:baseline_predicted.rfind(')') + 2]
        string_response += "< " + ', '.join([o[1:] for o in coref_objects if o != '']) + ">"
        string_response += " <EOB>"
        flatten_response.append(string_response)
        # dialogue_idx, turn = d_turn.split(':')
        # act = content[content.find('=>')+ len('=>'): content.rfind('<SSV>')].strip()
        # slot_val = content[content.find('<SSV>')+ len('<SSV>'): content.rfind('<ESV>')].strip()
        # req_slot = content[content.find('<SRS>')+ len('<SRS>'): content.rfind('<ERS>')].strip()
        # o_coref = content[content.find('<SOCR>')+ len('<SOCR>'): content.rfind('<EOCR>')].strip()
        # if len(content.split('<EOCR>')) > 1:
        #  u_bracket = content.split('<EOCR>')[1].strip()
        # else:
        #     print('u bracket not found')
        #     u_bracket = '<>'
        # u_coref = u_bracket[u_bracket.find('<')+len('<'): u_bracket.rfind('>')].replace(' ','').replace('U', ' U').strip()
        #
        # dt_key = dialogue_idx + '_' +turn
        # scene_key = prompt_dict[dt_key]['scene']
        # if u_coref != '':
        #    coref_objects = predict_coref(u_coref, o_coref, unique2obj[scene_key+"_scene.json"], unique2salency[scene_key+"_scene.json"])
        # elif o_coref != '':
        #     coref_objects = []
        #     print('AAA',len(o_coref.split(' ')), o_coref.split(' '))
        #     coref_objects = o_coref.split(' ')
        # else:
        #     coref_objects = []
        #
        #
        # string_response +=  act + " "
        # string_response += format_sv_string(slot_val) + ' '
        # string_response += format_req_string(req_slot) + ' '
        # string_response += '< ' + ', '.join([obj.replace('O','') for obj in coref_objects]) + ' >'
        # string_response += ' <EOB>'
        # flatten_response.append(string_response)
        # print(string_response)

    with open(out_response, 'w+') as f:
        for item in flatten_response:
            f.write("%s\n" % item)
