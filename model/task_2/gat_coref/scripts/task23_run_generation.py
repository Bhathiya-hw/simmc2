#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

Adapted from
https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
"""

import argparse
import logging
import os
import numpy as np
import torch
import json
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    AutoConfig,

    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)
#from gat_gpt2.scripts.graph_representation.sg_data_entry import sg_feature_lookup
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
import gat_coref.scripts.graph_representation.sg_data_entry as sg_data_entry
import gat_coref.scripts.graph2dial as g2d

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "graph2dial": (g2d.Graph2Dial, GPT2Tokenizer),
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info(
            "CTRL typically works better with lower temperatures (and lower top_k)."
        )

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info(
            "WARNING! You are not starting your generation from a control code so you won't get good results"
        )
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input(
                    "Using XLM. Select language in "
                    + str(list(available_languages))
                    + " >>> "
                )

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (
        args.padding_text if args.padding_text else PADDING_TEXT
    ) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (
        args.padding_text if args.padding_text else PADDING_TEXT
    ) + prompt_text
    return prompt_text
tmp_text_list = []

def prepare_graph2dial_input(args,_,tokenizer, prompt_text,scene_key, belief_text):
        sg_datum = args.sg_feature.convert_one_gqa_scene_graph(scene_key.strip() + "_scene.json", tokenizer)
        belief_encoded = tokenizer.convert_tokens_to_ids(belief_text.split(","))
        belief_length = 2
        belief_encoded_processed =  belief_encoded + [0] * (belief_length - len(belief_encoded))

        encoded_prompt = tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
            return_tensors="pt",
            add_space_before_punct_symbol=True,
        )

        sg_datum_processed = torch_geometric.data.Batch.from_data_list([sg_datum])
        encoded_belief_input = torch.tensor([belief_encoded_processed])
        return encoded_prompt,sg_datum_processed,encoded_belief_input


def prepare_graph2dial_input_task23(args, _, tokenizer, prompt_dict, prompt):
    datum = args.sg_feature.convert_one_gqa_scene_graph2(prompt_dict['scene'] + "_scene.json", tokenizer)
    context = []
    if prompt_dict['turn_idx'] > 1:
        prev_turn_key = str(prompt_dict['dialog_idx']) + "_" + str(prompt_dict['turn_idx'] - 1)
        prev_dict = prompt[prev_turn_key]
        prev_datum = datum if prev_dict['scene'] == prompt_dict['scene'] else args.sg_feature.convert_one_gqa_scene_graph2(prev_dict['scene'] + "_scene.json", tokenizer)
        context += tokenizer.encode(prev_dict['prev_asst_utterance'], add_special_tokens=True, max_length=args.block_size)
        if 'visual_objects' in prev_dict.keys():
            unique_vis = ['['] + [prev_datum.local2unique[obj] for obj in prev_dict['visual_objects'].split(' ') if obj in list(prev_datum.local2unique.keys())] + [']']
            context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in
                        prev_dict['visual_objects'].split(' ')]  # tokenizer.convert_tokens_to_ids(v['visual_objects'],add_special_tokens=True, max_length=block_size)
            context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in unique_vis]
        context += tokenizer.encode(prev_dict['user_utterance'], add_special_tokens=True, max_length=args.block_size)

    if prompt_dict['turn_idx'] > 0:
        context += tokenizer.encode(prompt_dict['prev_asst_utterance'], add_special_tokens=True, max_length=args.block_size)
        if 'visual_objects' in prompt_dict.keys():
            unique_vis = ['['] + [datum.local2unique[obj] for obj in prompt_dict['visual_objects'].split(' ') if obj in list(datum.local2unique.keys())] + [']']
            context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in
                        prompt_dict['visual_objects'].split(' ')]  # tokenizer.convert_tokens_to_ids(v['visual_objects'],add_special_tokens=True, max_length=block_size)
            context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in unique_vis]
    context += tokenizer.encode(prompt_dict['user_utterance'], add_special_tokens=True, max_length=args.block_size)
    context += tokenizer.encode('=>', add_special_tokens=True, max_length=args.block_size)
    return torch.tensor([context], dtype=torch.long), torch_geometric.data.Batch.from_data_list([datum])

def prepare_graph2dial_input_task23_new(args, _, tokenizer, prompt_dict, prompt):
    datum = args.sg_feature.convert_one_gqa_scene_graph2(prompt_dict['scene'] + "_scene.json", tokenizer)
    context = ''
    if prompt_dict['turn_idx'] > 1:
        prev_turn_key = str(prompt_dict['dialog_idx']) + "_" + str(prompt_dict['turn_idx'] - 1)
        prev_dict = prompt[prev_turn_key]
        prev_datum = datum if prev_dict['scene'] == prompt_dict['scene'] else args.sg_feature.convert_one_gqa_scene_graph2(prev_dict['scene'] + "_scene.json", tokenizer)
        context += prev_dict['prev_asst_utterance'] #tokenizer.encode(prev_dict['prev_asst_utterance'], add_special_tokens=True, max_length=args.block_size)
        if 'visual_objects' in prev_dict.keys():
            unique_vis = [prev_datum.local2unique[obj] for obj in prev_dict['visual_objects'].split(' ') if obj in list(prev_datum.local2unique.keys())]
            context +=  prev_dict['visual_objects']
            # context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in
            #             prev_dict['visual_objects'].split(' ')]  # tokenizer.convert_tokens_to_ids(v['visual_objects'],add_special_tokens=True, max_length=block_size)
            context += "[ " + ",".join(unique_vis) + " ]"
            # context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in unique_vis]
        context += prev_dict['user_utterance'] #tokenizer.encode(prev_dict['user_utterance'], add_special_tokens=True, max_length=args.block_size)

    if prompt_dict['turn_idx'] > 0:
        context += prompt_dict['prev_asst_utterance']#tokenizer.encode(prompt_dict['prev_asst_utterance'], add_special_tokens=True, max_length=args.block_size)
        if 'visual_objects' in prompt_dict.keys():
            unique_vis = [datum.local2unique[obj] for obj in prompt_dict['visual_objects'].split(' ') if obj in list(datum.local2unique.keys())]
            context += prompt_dict['visual_objects']
            context += "[ " + ",".join(unique_vis) + " ]"
            # context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in
            #             prompt_dict['visual_objects'].split(' ')]  # tokenizer.convert_tokens_to_ids(v['visual_objects'],add_special_tokens=True, max_length=block_size)
            # context += [extended_tokenizer_encode(tokenizer=tokenizer, token=tk.strip(), block_size=args.block_size) for tk in unique_vis]
    context += prompt_dict['user_utterance']#tokenizer.encode(prompt_dict['user_utterance'], add_special_tokens=True, max_length=args.block_size)
    context += ' => ' #tokenizer.encode('=>', add_special_tokens=True, max_length=args.block_size)
    context_encoded = tokenizer.batch_encode_plus([context], add_special_tokens=True, max_length=args.block_size)["input_ids"]
    return torch.tensor(context_encoded, dtype=torch.long), torch_geometric.data.Batch.from_data_list([datum])

def extended_tokenizer_encode(tokenizer, token, block_size):
    converted_id = tokenizer.convert_tokens_to_ids(token)
    return converted_id if converted_id != 50256 else tokenizer.encode(token, add_special_tokens=True, max_length=block_size)


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
    "graph2dial": prepare_graph2dial_input_task23_new,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    #@TODO move to config
    parser.add_argument(
        "--with_ins",
        action="store_true",
        help="gat conv with instruction or without",
    )

    parser.add_argument(
        "--gat_conv_layers",
        default=None,
        type=int,
        required=True,
        help="gat conv with instruction or without",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--prompts_from_file",
        type=str,
        default=None,
        help="""
read prompts from a file and generate, overrides any prompt given on the
command line""",
    )
    parser.add_argument(
        "--graph_json_file",
        type=str,
        default=None,
        help="""
    read graphs from a file and generate, overrides any prompt given on the
    command line""",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        default=None,
        help="""
    read prompts from a file and generate, overrides any prompt given on the
    command line""",
    )
    parser.add_argument(
        "--belief_file",
        type=str,
        default=None,
        help="""
    read scenes from a file and generate""",
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Padding text for Transfo-XL and XLNet.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Path to output predictions in a line separated text file.",
    )
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    # torch.cuda.set_device(2)
    args.n_gpu = 0 if args.no_cuda else 1 #torch.cuda.device_count()

    set_seed(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained( pretrained_model_name_or_path= args.model_name_or_path, config=config, tokenizer = tokenizer, with_ins=args.with_ins, gat_conv_layers=args.gat_conv_layers)
    model.to(args.device)

    args.sg_feature = sg_data_entry.sg_feature_lookup(args.graph_json_file,tokenizer)
    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    args.block_size = args.length
    logger.info(args)

    results = []
    prompts = []
    if args.prompts_from_file:
        with open(args.prompts_from_file) as handle:
            prompts = json.load(open(args.prompts_from_file))#handle.readlines()
            # prompts = {k: prompts[k] for k in list(prompts)[:4]}
    if args.scene_file:
        with open(args.scene_file) as scene:
            scene_keys = scene.readlines()
    if args.belief_file:
        with open(args.belief_file) as belief:
            beliefs = belief.readlines()
    while True:
        if not prompts:
            prompts = [args.prompt if args.prompt else input("Model prompt >>> ")]
            if not args.prompt and (
                len(prompts) == 0
                or prompts[0].strip() == ""
                or prompts[0].lower() == "quit"
            ):
                break  # break while True loop

        n_prompts = len(prompts)
        i = 0
        for prompt_key, prompt_dict in prompts.items():
            i += 1
            # Strip any trailing \n if provided
            # prompt_text = prompt_text.strip("\n")

            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                encoded_prompt, encoded_sg_input = prepare_input(
                    args, model, tokenizer, prompt_dict,prompts )

            encoded_prompt = encoded_prompt.to(args.device)
            encoded_sg_input =encoded_sg_input.to(args.device)

            # instantiate logits processors
            # logits_processor = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(0.1)])
            # instantiate logits processors
            # logits_warper = LogitsProcessorList([TemperatureLogitsWarper(0.7),TopKLogitsWarper(top_k=
            logits_processor = LogitsProcessorList([])
            logits_warper = LogitsProcessorList([TopKLogitsWarper(top_k=1)])
            output_sequences = model.sample(
                    input_ids = encoded_prompt,
                    logits_processor = logits_processor,
                    stopping_criteria= None,
                    logits_warper = logits_warper,
                    max_length = encoded_prompt.shape[1]+ 50,
                    # pad_token_id = 50256,
                    # eos_token_id = 50256,
                    output_attentions = False,
                    output_hidden_states = False,
                    output_scores = False,
                    return_dict_in_generate = False,
                    sg_input = encoded_sg_input,
                    )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []

            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                        sequence_idx=generated_sequence_idx + 1,
                        promt_idx=i,
                        n_prompts=n_prompts,
                    )
                )
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the
                # excess text that was used for pre-processing
                total_sequence = (
                    str(prompt_dict)
                    + text[
                        len(
                            tokenizer.decode(
                                encoded_prompt[0], clean_up_tokenization_spaces=True
                            )
                        ) :
                    ]
                )

                generated_sequences.append(total_sequence)
                print(total_sequence)
                result = str(prompt_dict['dialog_idx']) +":"+ str(prompt_dict['turn_idx']) + "<=>" + tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # results.append(generated_sequences)
            results.append(result)

        prompts = []
        if args.prompt or args.prompts_from_file:
            break  # break while True loop

    if args.path_output is not None:

        # Create a directory if it does not exist
        directory = os.path.dirname(args.path_output)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Format results into a line-separated string file
        str_results = "\n".join(
            results
        )

        # Save to a file
        with open(args.path_output, "w") as f_out:
            f_out.write(str_results)

    return results


if __name__ == "__main__":
    main()
