#!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.

Adapted from:
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py
"""


import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
import torch.nn.functional as F
# import sys

# sys.path.append("./../../")
import glimmer_coref.scripts.graph_representation.sg_data_entry as sg_data_entry
import glimmer_coref.scripts.graph_representation.Constants as Constants
import glimmer_coref.scripts.glimmer as g2d
# import  gat_coref.scripts.graph_representation.gat_encoder_decoder as enc_dec
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512
    ):
        assert os.path.isfile(file_path)
        block_size = block_size - (
            tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        print(("block size", block_size))
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        for i in range(
            0, len(tokenized_text) - block_size + 1, block_size
        ):  # Truncate in block of block_size
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, predict_file_path:str, scene_file_path :str, belief_file_path:str, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(belief_file_path, encoding="utf-8") as f:
            beliefs = [belief for belief in f.read().splitlines() if (len(belief) > 0 and not belief.isspace()) ]
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(predict_file_path, encoding="utf-8") as f:
            predict_lines = [predict_line for predict_line in f.read().splitlines() if (len(predict_line) > 0 and not predict_line.isspace())]

        # scene_graph_load = json.load(open(self.graph_json_path))
        with open(scene_file_path, encoding='utf-8') as f:
            scene_lines = [scene_line for scene_line in f.read().splitlines() if (len(scene_line) > 0 and not scene_line.isspace())]

        # scene_graphs_keys = [scene_graph_load[scene_line + '_scene.json'] for scene_line in scene_lines ]
        scene_data_dict = {}
        sg_data = []
        belief_encoded_final = []
        if args.with_ins:
            belief_encoded = []
            for belief in beliefs:
                belief_encoded.append(tokenizer.convert_tokens_to_ids(belief.split(",")))
            belief_length = args.gat_conv_layers
            belief_encoded_final  = [ seq + [0] * (belief_length -len(seq)) if len(seq)<belief_length  else seq[:belief_length ] for seq in belief_encoded ]

        for key in scene_lines:
            if key not in [str(k) for k in scene_data_dict.keys()]:
                datum = args.sg_feature.convert_one_gqa_scene_graph2(key + '_scene.json', tokenizer)
                sg_data.append(datum)
                scene_data_dict[key] = datum
            else:
                sg_data.append(scene_data_dict[key].clone())

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.predict_lines = tokenizer.batch_encode_plus(predict_lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.scenes = sg_data
        self.belief_tokens = belief_encoded_final
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.belief_tokens:
            return torch.tensor(self.examples[i], dtype=torch.long) ,torch.tensor(self.predict_lines[i], dtype=torch.long), self.scenes[i], torch.tensor(self.belief_tokens[i], dtype=torch.long)
        else:
            return torch.tensor(self.examples[i], dtype=torch.long),torch.tensor(self.predict_lines[i], dtype=torch.long), self.scenes[i], torch.tensor([[0]])
def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    scene_file_path = args.scene_eval_data_file if evaluate else args.scene_train_data_file
    belief_file_path = args.belief_eval_data_file if evaluate else args.belief_train_data_file
    predict_file_path = args.predict_eval_data_file if evaluate else args.predict_train_data_file
    if args.line_by_line:
        dataset = LineByLineTextDataset(
            tokenizer, args, file_path=file_path, predict_file_path= predict_file_path, scene_file_path=scene_file_path, belief_file_path= belief_file_path, block_size=args.block_size
        )
    else:
        dataset = TextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )

    # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
    # Below is a workaround in case you encounter this issue.
    # Alternatively, --nocuda could avoid this issue too.
    # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    n = len(dataset) % args.per_gpu_train_batch_size
    if n != 0:
        print("Truncating from %d examples" % len(dataset.examples))
        dataset.examples = dataset.examples[:-n]
        dataset.scenes = dataset.scenes[:-n]
        print("Truncating to %d examples" % len(dataset.examples))
    return dataset

class MiniDictDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, context_file_path: str, block_size=512):

        assert os.path.isfile(context_file_path)
        # assert os.path.isfile(scene_file_path)

        with open(context_file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # with open(predict_file_path, encoding="utf-8") as f:
        #     pred_lines = [pred_line for pred_line in f.read().splitlines() if (len(pred_line) > 0 and not pred_line.isspace())]

        scene_data_dict = {}
        # with open(scene_file_path, encoding='utf-8') as f:
        #     scene_lines = [scene_line for scene_line in f.read().splitlines() if (len(scene_line) > 0 and not scene_line.isspace())]

        # sg_data = []
        # for key in scene_lines:
        #     if key not in [str(k) for k in scene_data_dict.keys()]:
        #         datum = args.sg_feature.convert_one_gqa_scene_graph2(key, tokenizer)
        #         sg_data.append(datum)
        #         scene_data_dict[key] = datum
        #     else:
        #         sg_data.append(scene_data_dict[key].clone())

        if args.special_encode_uniques:
            # self.prediction = self.special_encode(pred_lines,tokenizer)
            self.context = self.special_encode(lines, tokenizer)
        else:
            # self.prediction = tokenizer.batch_encode_plus(pred_lines, add_special_tokens=True, max_length=block_size)["input_ids"]
            self.context = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        # self.scene_data = sg_data

    def __len__(self):
        return len(self.context)

    def __getitem__(self, i):
        return torch.tensor(self.context[i], dtype=torch.long)#,torch.tensor(self.prediction[i], dtype=torch.long), self.scene_data[i]

    def special_encode(self,text_contents, tokenizer):
        encoded_lines = []
        for line in text_contents:
            line_encode = []
            if "Belief State :" in line:
                context, answer = line[:line.rfind("Belief State :") + len("Belief State :")], line[line.find("Belief State :") + len("Belief State :"):]
            else:
                context, answer = line[:line.rfind("Response:") + len("Response:")], line[line.find("Response:") + len("Response:"):]
            context_split = re.split('(<SCAT> | <ECAT>)', context)

            for idx, context_content in enumerate(context_split):
                if idx % 4 == 0:
                    multimedia_split = re.split('(<SOM> | <EOM>)', context_content.strip())
                    for inner_idx,mm_content in enumerate(multimedia_split):
                        if '<SREL>' in mm_content:
                            rel_split = re.split('(<SREL> | <EREL>)', mm_content)
                            for rindex, r in enumerate(rel_split):
                                if rindex %4 !=2:
                                    line_encode += tokenizer.encode(r)
                                else:
                                    relations = r.split(',')
                                    for rel_ind,rel in enumerate(relations):
                                        line_encode += [tokenizer.convert_tokens_to_ids(token) for token in rel.split(' ') if token != '']
                                        if rel_ind < len(relations)-1:
                                            line_encode += tokenizer.encode(',')
                        else:
                            if inner_idx % 4 != 2:
                                line_encode += tokenizer.encode(mm_content)
                            else:
                                line_encode += [tokenizer.convert_tokens_to_ids(token) for token in mm_content.split(', ') if token != '']

                elif idx % 4 != 2:
                    line_encode += tokenizer.encode(context_content)
                    # print(line_encode)
                else:
                    line_encode += [tokenizer.convert_tokens_to_ids(token) for token in context_content.split(', ') if token != '']
                    # print(line_encode)
            # print(line_encode)
            if "Belief State :" in line:
                answer = ''  if  answer == ' ' else answer
                answer_split = re.split('(<SPCT> | <EPCT>)', answer)
                for idx, answer_content in enumerate(answer_split):
                    if idx == 0 and answer_content != '':
                        act =  re.split('(\[ | \] )', answer_content.strip())[0]
                        line_encode += [tokenizer.convert_tokens_to_ids(act.strip())]

                        slot_groups = re.split('(\[ | \] )', answer_content.strip())[1:4]
                        line_encode += [tokenizer.convert_tokens_to_ids(slot_groups[0].strip())]
                        line_encode += tokenizer.encode(slot_groups[1])
                        line_encode += [tokenizer.convert_tokens_to_ids(slot_groups[2].strip())]

                        req_groups = re.split('(\(|\) )', answer_content.strip())[1:4]
                        line_encode += [tokenizer.convert_tokens_to_ids(req_groups[0].strip())]
                        line_encode += [tokenizer.convert_tokens_to_ids(token) for token in req_groups[1].split(', ') if token != '']
                        line_encode += [tokenizer.convert_tokens_to_ids(req_groups[2].strip())]

                        o_groups = re.split('(< | >)', answer_content.strip())[1:4]
                        line_encode += [tokenizer.convert_tokens_to_ids(o_groups[0].strip())]
                        line_encode += [tokenizer.convert_tokens_to_ids(token) for token in o_groups[1].split(', ') if token != '']
                        line_encode += [tokenizer.convert_tokens_to_ids(o_groups[2].strip())]

                    elif idx % 4 != 2:
                        line_encode += tokenizer.encode(answer_content)
                        # print(line_encode)
                    else:
                        line_encode += [tokenizer.convert_tokens_to_ids(token) for token in answer_content.split(', ') if token != '']
            else:
                line_encode += tokenizer.encode(answer)
            print(tokenizer.decode(line_encode))
            encoded_lines.append(line_encode)
        return encoded_lines

def load_and_cache_for_task2(args, tokenizer, evaluate=False):
    # scene_file_path = args.scene_eval_data_file if evaluate else args.scene_train_data_file
    context_file_path = args.eval_data_file if evaluate else args.train_data_file
    # pred_file_path = args.predict_eval_data_file if evaluate else args.predict_train_data_file

    dataset = MiniDictDataset(args= args, tokenizer=tokenizer, context_file_path=context_file_path, block_size = args.block_size)

       # Unknown issues have been reported around not being able to handle incomplete batches (e.g. w/ older CUDA 9.2)
       # Below is a workaround in case you encounter this issue.
       # Alternatively, --nocuda could avoid this issue too.
       # Comment out the following if you do not encounuter this issue or if you are not using any GPU.
    n = len(dataset) % args.per_gpu_train_batch_size
    if n != 0:
        print("Truncating from %d examples" % len(dataset.context))
        dataset.context = dataset.context[:-n]
        # dataset.acts = dataset.acts[:-n]
        # dataset.slot_values = dataset.slot_values[:-n]
        # dataset.request_slots = dataset.request_slots[:-n]
        # dataset.scene_data = dataset.scene_data[:-n]
        # dataset.answers = dataset.answers[:-n]
        print("Truncating to %d examples" % len(dataset.context))
    return dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)

def label_tokens (sg_input, conv_labels):
    dense_x = torch_geometric.utils.to_dense_batch(sg_input.x, batch=sg_input.batch)[0]
    conv_labels_new = F.pad(conv_labels, pad=(dense_x.shape[1], 0), value=0)
    return conv_labels_new

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(data):
        context = data
        return pad_sequence(context, batch_first=False, padding_value=0)
        # context, predict, sg_datum = tuple(zip(*data))
        # sg_datum = torch_geometric.data.Batch.from_data_list(sg_datum)
        # return pad_sequence(context, batch_first=False, padding_value=0),\
        #        pad_sequence(predict, batch_first=False, padding_value=0),\
        #        sg_datum,\


    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    # model.resize_token_embeddings(len(tokenizer)) @TODO Check

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            # torch.cuda.empty_cache()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            target = batch
            labels = target.clone()
            # labels = torch.cat((torch.full_like(sg_input.x[:, 0].unsqueeze(1), fill_value=-100), target))

            target = target.to(args.device)
            labels = labels.to(args.device)
            model.train()

            outputs = (
                   model(input_ids = target.T, labels =labels.T)
            )


            for i, output in enumerate(outputs[1]):
                print("Input : " + (''.join(token for token in tokenizer.convert_ids_to_tokens(target.T[i]))).replace('Ġ', " "))
                print("Output : " + (''.join(token for token in tokenizer.convert_ids_to_tokens(torch.argmax(output, dim=1))).replace('Ġ', " ")))

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # loss = outputs
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate( args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix=""):
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_for_task2(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    def collate(data):
        context, predict, sg_datum = tuple(zip(*data))
        sg_datum = torch_geometric.data.Batch.from_data_list(sg_datum)
        return pad_sequence(context, batch_first=False, padding_value=0),\
               pad_sequence(predict, batch_first=False, padding_value=0),\
               sg_datum,\

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        target, predict, sg_input, = batch
        labels = torch.cat((torch.full_like(sg_input.x[:, 0].unsqueeze(1), fill_value=-100), target))

        target = target.to(args.device)
        labels = labels.to(args.device)
        predict = predict.to(args.device)
        sg_input = sg_input.to(args.device)

        with torch.no_grad():
            outputs = (
                # model(questions = context, gt_scene_graphs = sg_input, programs_input = slot_values, short_answers = None, full_answers = answer, act = act, slot_values = slot_values,request_slots = request_slots )
                model(input_ids=target, predict_input_ids=predict, sg_input=sg_input, labels=labels)
            )
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )

    parser.add_argument(
        "--special_encode_uniques",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )

    parser.add_argument(
        "--should_continue",
        action="store_true",
        help="Whether to continue from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=1000, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

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
    args = parser.parse_args()

    if (
        args.model_type in ["bert", "roberta", "distilbert", "camembert"]
        and not args.mlm
    ):
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError(
                "Used --should_continue but no checkpoint was found in --output_dir."
            )
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # torch.backends.cudnn.enabled = False
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # torch.cuda.set_device(2)
        args.n_gpu = 0 if args.no_cuda else 1 #torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, cache_dir=args.cache_dir
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    # tokenizer.bos_token = '<SOCR>'
    # args.sg_feature = sg_data_entry.sg_feature_lookup(args.graph_json_file,tokenizer)

    if args.add_special_tokens:
        print(args.add_special_tokens)
        if not os.path.exists(args.add_special_tokens):
            raise ValueError(
                "Additional special tokens file {args.add_special_tokens} not found}"
            )
        with open(args.add_special_tokens, "rb") as handle:
            special_tokens_dict = json.load(handle)
        tmp_text_list = []

        tmp_text_list += Constants.RELATIONS_INV + Constants.ATTRIBUTES_INV
        tmp_text_list.append("<self>")  # add special token for self-connection
        # tmp_text_list = [tmp_text_list]
        for txt in tmp_text_list:
            if txt not in special_tokens_dict['additional_special_tokens']:
                special_tokens_dict['additional_special_tokens'].append(txt)
        # special_tokens_dict['additional_special_tokens'].extend(tmp_text_list)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")

        # args.mini_dict = {k:tokenizer.vocab[k] for k in tmp_text_list}

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        print(args.block_size)
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
            model = g2d.Graph2Dial(config=config, pretrained_model_path=args.model_name_or_path, cache_dir=args.cache_dir , tokenizer=tokenizer, add_special_tokens=args.add_special_tokens, with_ins=args.with_ins, gat_conv_layers=args.gat_conv_layers)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    # ensure model aligns with any addition of special tokens
    # (unclear if this step is needed for a new model)
    # if args.add_special_tokens:
    #     model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_for_task2(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            ]
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = g2d.Graph2Dial.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=config, tokenizer=tokenizer, with_ins=args.with_ins, gat_conv_layers=args.gat_conv_layers)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = {k + "_{}".format(global_step): v for k, v in result.items()}
            results.update(result)

    return results


if __name__ == "__main__":
    main()
