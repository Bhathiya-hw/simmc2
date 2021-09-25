from copy import deepcopy

import torch

from torch.nn import Sequential, Linear, ReLU, Embedding, LayerNorm, Dropout, ModuleList
from torch_scatter import scatter_mean, scatter_add
import logging
import torch_geometric
import gat_coref.scripts.graph_representation.sg_encoder as sg_encoder
import gat_coref.scripts.graph_representation.gat_conv as gat_conv
import gat_coref.scripts.graph_representation.gat_encoder_decoder as enc_dec
import torch.nn.functional as F

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup, add_start_docstrings,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2PreTrainedModel
)

from  transformers.generation_logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper
from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
logger = logging.getLogger(__name__)

class Graph2Dial(GPT2PreTrainedModel):
    def __init__(self, config,tokenizer, pretrained_model_path=None, cache_dir= None ,add_special_tokens =True, with_ins =False, gat_conv_layers=5 ):
        super(Graph2Dial, self).__init__(config)
        self.tokenizer = tokenizer
        if pretrained_model_path:
            self.transformer =AutoModelWithLMHead.from_pretrained(
                        pretrained_model_path,
                        from_tf=bool(".ckpt" in pretrained_model_path),
                        config=config,
                        cache_dir=cache_dir,
                    )
        else:
            self.transformer = AutoModelWithLMHead.from_config(config)
        if add_special_tokens:
             self.transformer.resize_token_embeddings(len(tokenizer))

        self.with_ins = with_ins
        self.ins_dim = self.transformer.transformer.wte.embedding_dim
        self.encoder_decoder = enc_dec.GATEncoderDecoder(tokenizer=tokenizer, gat_conv_layers=gat_conv_layers, ins_dim=self.ins_dim, question_hidden_dim=self.ins_dim)
        self.binary_classifier = torch.nn.Linear(self.ins_dim,2)

        out_classifier_dim = 768
        # self.logit_fc = torch.nn.Linear(out_classifier_dim, 1)
        self.direct_fc = torch.nn.Sequential(
            torch.nn.Linear(self.ins_dim, self.ins_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.ins_dim),
            torch.nn.Linear(self.ins_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.predicted_fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.ins_dim),
            torch.nn.Linear(self.ins_dim, self.ins_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.ins_dim),
            torch.nn.Linear(self.ins_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(out_classifier_dim, 1)
        )
        # self.question_encoder = enc_dec.TransformerQuestionEncoder()

        self.direct_ans_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
        self.predicted_ans_criterion = torch.nn.BCEWithLogitsLoss()
        # self.scene_graph_encoder = sg_encoder.sg_encoder(len(tokenizer), self.ins_dim, tokenizer.pad_token_id)

        # self.gat_seq = gat_conv.gat_seq(in_channels=self.scene_graph_encoder.sg_emb_dim,
        #                    out_channels=self.scene_graph_encoder.sg_emb_dim,
        #                    edge_attr_dim=self.scene_graph_encoder.sg_emb_dim,
        #                    with_ins= self.with_ins,
        #                    ins_dim=512, gat_conv_layers=gat_conv_layers,
        #                    dropout=0.1, gat_heads=4, gat_negative_slope=0.2, gat_bias=True)  # the drop-out is for both dropout in

        # self.graph_global_attention_pooling = MyConditionalGlobalAttention(
        #     num_node_features=self.scene_graph_encoder.sg_emb_dim,
        #     num_out_features=512)
        # self.tie_lm_weights()

    def classification_target(self,x_encoded, x_executed, sg_input, answer):
        yes_token = self.tokenizer.encode("YES")
        # yes_embedding = self.transformer.transformer.wte(yes_token)
        no_token = self.tokenizer.encode("NO")
        # no_embedding = self.transformer.transformer.wte(no_token)
        eos = torch.tensor([self.tokenizer.eos_token_id])
        eos_token_embed = self.transformer.transformer.wte(torch.tensor([eos]))

        target_classification = torch.tensor([yes_token if sg_input.x[:, 0][i].item() in answer else no_token for i in range(sg_input.x.shape[0])]).to(answer.device)

        target_classification_embedding = self.transformer.transformer.wte(target_classification)
        target_classification_embedding = torch.stack(((x_encoded.unsqueeze(1), x_executed.unsqueeze(1), target_classification_embedding)))
        target_classification_embedding = torch.flatten(target_classification_embedding.transpose(0,1), start_dim=0, end_dim=1)
        target_classification_embedding = torch.cat((target_classification_embedding, eos_token_embed.unsqueeze(0)))

        label_filling = torch.full((x_encoded.shape[0], 2), fill_value=-100).to(answer.device)
        target_classification = torch.flatten(torch.cat((label_filling,target_classification),dim=1)).unsqueeze(1)
        target_classification = torch.cat((target_classification, eos.unsqueeze(0)))

        return target_classification_embedding,target_classification
        # input = torch.stack((x_encoded,x_executed,target_classification_embedding))
        # label = torch.cat((label_filling,target_classification))

    def forward(self, input_ids, predict_input_ids, sg_input=None,  labels=None, return_dict=False,output_attentions=None,
                    output_hidden_states=None, act = None, answer = None, request_slots = None, slot_values = None, past_key_values=None, attention_mask=None, token_type_ids=None,
                            position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                            encoder_attention_mask=None, use_cache=None):


        # print("answer :", self.tokenizer.decode(input_ids[predict_input_ids.shape[0]+1:-2][:,0]))
        # target_classification = torch.tensor([1.0 if sg_input.x[:, 0][i].item() in answer else 0.0 for i in range(sg_input.x.shape[0])]).to(sg_input.x.device)
        # direct answer loss calculation
        question_encoded = self.transformer.transformer.wte(predict_input_ids.T)
        x_encoded, x_executed, edge_attn, sg_embeds = self.encoder_decoder(questions=question_encoded,
                                                                gt_scene_graphs=sg_input,
                                                                programs_input=slot_values,
                                                                short_answers=None,
                                                                full_answers=answer,
                                                                act=act,
                                                                slot_values=slot_values,
                                                                request_slots=request_slots)

        # conv_input_embed = self.transformer.transformer.wte(input_ids)
        # new_inputs, new_labels =  self.classification_target(x_encoded,x_executed, sg_input,answer)
        # new_input_embeds = torch.cat((question_encoded, new_inputs))
        # new_input_labels = torch.cat((labels[:predict_input_ids.shape[0]], new_labels))
        # dial_out = self.transformer(inputs_embeds=new_input_embeds.transpose(1, 0), labels=new_input_labels.T, return_dict=True, output_attentions=output_attentions, output_hidden_states=True)

        #[[self.tokenizer.decode(sg_input.x[:,0][i]) for i in range(sg_input.x.shape[0])]]

        # question_node_relevance = torch.mean(torch.nn.functional.softmax(torch.matmul(question_encoded.squeeze(1), x_executed.transpose(1, 0)), dim=1),dim=0)
        # print(','.join([self.tokenizer.decode(sg_input.x[k][0]) for k in question_node_relevance.topk(5)[1]]))
        # da_loss = self.direct_ans_criterion(question_node_relevance, target_classification)
        # direct_logits = self.direct_fc(x_executed)
        # print([(self.tokenizer.decode(sg_input.x[i][0]), torch.nn.functional.sigmoid(direct_logits[i])) for i in range(direct_logits.shape[0])])
        # da_loss = self.direct_ans_criterion(direct_logits.squeeze(1), target_classification)

        conv_input_embed = self.transformer.transformer.wte(input_ids)
        inputs_embeds = torch.cat((x_executed.unsqueeze(0), conv_input_embed), dim=1) #fine_tune -1
        # inputs_embeds = torch.cat((x_encoded.unsqueeze(1), conv_input_embed), dim=0)
        # dial_out = self.transformer(inputs_embeds=inputs_embeds.transpose(1,0), labels=labels.T, return_dict=True,output_attentions=output_attentions, output_hidden_states=True)
        dial_out = self.transformer(inputs_embeds=inputs_embeds, return_dict=True, output_attentions=output_attentions, output_hidden_states=True)

        # final_layer = dial_out['hidden_states'][12]
        # prediction_relevance = torch.mean(torch.nn.functional.softmax(torch.matmul(final_layer.squeeze(0), x_executed.transpose(1, 0)), dim=1),dim=0)
        # print(','.join([self.tokenizer.decode(sg_input.x[k][0]) for k in prediction_relevance.topk(5)[1]]))

        # pred_loss = self.predicted_ans_criterion(prediction_relevance, target_classification)
        #
        #
        # print("da_loss:{}  dial_out: {}".format(da_loss, dial_out['loss']))
        # loss = da_loss + pred_loss + 0.1 * dial_out['loss']
        # loss = dial_out['loss']
        #GAT
        #sg_input = torch_geometric.data.Batch.from_data_list(sg_input).to(input_ids.device)
        # x_encoded, edge_attr_encoded,_ = self.scene_graph_encoder(sg_input)
        # if self.with_ins:
        #     mem = self.transformer.transformer.wte(belief_input.T)
        # else:
        #     mem = None
        # x_executed = self.gat_seq(x=x_encoded, edge_index=sg_input.edge_index, edge_attr=edge_attr_encoded, instr_vectors=mem, batch=sg_input.batch)
        #print(("sg_input batch", sg_input.batch.shape, sg_input.batch.device, sg_input.ptr))
        #print(("inputs_ids", input_ids.shape, input_ids.device))
        # print(x_executed.shape, sg_input.batch.shape)
        # dense_x_executed = torch_geometric.utils.to_dense_batch(x_executed,batch=sg_input.batch.clone())[0]
        #conv_labels_new = F.pad(labels, pad=(dense_x_executed.shape[1], 0), value=-100),

        #Concat GAT+GPT2 Input
        # conv_input_embed = self.transformer.transformer.wte(input_ids)

        # question_encoded = self.transformer.transformer.wte(predict_input_ids)


        # sg_embeds = self.encoder_decoder(questions= predict_input_ids, gt_scene_graphs = sg_input, programs_input = None, full_answers = None,
        #                       short_answers = None)#torch.cat((dense_x_executed, conv_input_embed ),dim =1)
        #[[self.tokenizer.decode(sg_input.x[:, i][j]) for i in range(sg_input.x.shape[1])] for j in range(sg_input.x.shape[0])]



        # inputs_embeds = torch.cat((sg_embeds, conv_input_embed), dim=0)

        #GPT2
        # dial_out = self.transformer(inputs_embeds=inputs_embeds.transpose(1,0), labels=labels.T,return_dict=return_dict,output_attentions=output_attentions,
        #                output_hidden_states=output_hidden_states)
        # dial_out = self.transformer(input_ids=input_ids, labels=input_ids,return_dict=True,output_attentions=output_attentions,
        #                 output_hidden_states=output_hidden_states)
        # print(torch.argmax(torch.nn.functional.softmax(self.transformer(input_ids=input_ids[:, :50], return_dict=True, output_attentions=output_attentions,
        #                                                            output_hidden_states=output_hidden_states)['logits'][:, -1, :]), dim=1))
        # print(loss)
        # print_labels = new_input_labels.detach().clone()
        # print_labels[print_labels == -100] = 0
        # for i, output in enumerate(dial_out[1]):
        #     print("Input : " + (''.join(token for token in self.tokenizer.convert_ids_to_tokens(print_labels.T[i]))).replace('Ġ', " "))
        #     print("Output : " + (''.join(token for token in self.tokenizer.convert_ids_to_tokens(torch.argmax(output, dim=1))).replace('Ġ', " ")))
        # del print_labels, loss
        return dial_out#, da_loss,pred_loss, dial_out

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor = None,
        stopping_criteria= None,
        logits_warper = None,
        max_length = 100,
        pad_token_id = 50256,
        eos_token_id = 50256,
        output_attentions = False,
        output_hidden_states = False,
        output_scores = False,
        return_dict_in_generate = False,
        synced_gpus = False,
        **model_kwargs,
    ) :
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.SampleDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.SampleEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForCausalLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    TopKLogitsWarper,
            ...    TemperatureLogitsWarper,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # if max_length is not None:
            # warnings.warn(
            #     "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            #     UserWarning,
            # )
        stopping_criteria = self.validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # if synced_gpus and this_peer_finished:
            #     cur_len = cur_len + 1
            #     continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                # this_peer_finished = True
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return SampleEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return SampleDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        return input_ids

    def validate_stopping_criteria(self, stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
        stopping_max_length = stopping_criteria.max_length
        new_stopping_criteria = deepcopy(stopping_criteria)
        if stopping_max_length is not None and stopping_max_length != max_length:
              print("You set different `max_length` for stopping criteria and `max_length")
        #     warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
        elif stopping_max_length is None:
            new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        return new_stopping_criteria

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        # if past:
        #     input_ids = input_ids[:, -1].unsqueeze(-1)
        #     if token_type_ids is not None:
        #         token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        sg_input = kwargs.get("sg_input")
        belief_input = kwargs.get("belief_input")

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sg_input": sg_input,
            "predict_input_ids": input_ids
        }