from copy import deepcopy

import torch

from torch.nn import Sequential, Linear, ReLU, Embedding, LayerNorm, Dropout, ModuleList
from torch_scatter import scatter_mean, scatter_add
#from gat_gpt2.scripts.graph_representation.modified_graph_layernorm import LayerNorm as modifiedLayerNorm
import logging
import torch_geometric
#from gat_gpt2.scripts.graph_representation.sg_encoder import sg_encoder
#from gat_gpt2.scripts.graph_representation.gat_conv import gat, gat_seq
import gat_gpt2.scripts.graph_representation.sg_encoder as sg_encoder
import gat_gpt2.scripts.graph_representation.gat_conv as gat_conv
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

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from  transformers.generation_logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper
from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
# from  transformers.generation_utils import SampleDecoderOnlyOutput,SampleEncoderDecoderOutput
logger = logging.getLogger(__name__)

class MyConditionalGlobalAttention(torch.nn.Module):
    r"""Language-Conditioned Global soft attention layer

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( u[batch] ) \dot h_{\mathbf{\Theta}} ( \mathbf{x}_n ) \right)
        \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),
    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)

    """
    def __init__(self, num_node_features, num_out_features):
        super(MyConditionalGlobalAttention, self).__init__()
        channels = num_out_features
        self.gate_nn = Sequential(Linear(channels, channels), ReLU(), Linear(channels, 1))
        self.node_nn = Sequential(Linear(num_node_features, channels), ReLU(), Linear(channels, channels))
        self.ques_nn = Sequential(Linear(channels, channels), ReLU(), Linear(channels, channels))
        # self.gate_nn = Lin(channels, 1)
        # self.node_nn = Lin(channels, channels)
        # self.nn = Lin(num_node_features, channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)
        torch_geometric.nn.inits.reset(self.ques_nn)

    def forward(self, x, u, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        # gate = self.gate_nn(x).view(-1, 1)

        ##################################
        # Batch
        # shape: x - [ Num of Nodes, num_node_features] --> [ Num of Nodes, Feature Channels ]
        # shape: u - [ Batch Size, Feature Channels]
        # shape: u[batch] - [ Num of Nodes, Feature Channels]
        ##################################
        x = self.node_nn(x) # if self.node_nn is not None else x
        # print("x", x.size(), "u", u.size(), "u[batch]", u[batch].size())

        ##################################
        # torch.bmm
        # batch1 and batch2 must be 3D Tensors each containing the same number of matrices.
        # If batch1 is a b x n x m Tensor, batch2 is a b x m x p Tensor, out will be a b x n x p Tensor.
        ##################################


        gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        # gate = torch.bmm(x.unsqueeze(1) , self.ques_nn(u)[batch].unsqueeze(2)).squeeze(-1)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self):
        return '{}(gate_nn={}, node_nn={}, ques_nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.node_nn, self.ques_nn)


class Graph2Dial(GPT2PreTrainedModel):
    def __init__(self, config,tokenizer, pretrained_model_path=None, cache_dir= None ,add_special_tokens =True, with_ins =False, gat_conv_layers=5 ):
        super(Graph2Dial, self).__init__(config)
        if pretrained_model_path:
            self.transformer =AutoModelWithLMHead.from_pretrained(
                        "gpt2",
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
        self.scene_graph_encoder = sg_encoder.sg_encoder(len(tokenizer), self.ins_dim, tokenizer.pad_token_id)

        self.gat_seq = gat_conv.gat_seq(in_channels=self.scene_graph_encoder.sg_emb_dim,
                           out_channels=self.scene_graph_encoder.sg_emb_dim,
                           edge_attr_dim=self.scene_graph_encoder.sg_emb_dim,
                           with_ins= self.with_ins,
                           ins_dim=512, gat_conv_layers=gat_conv_layers,
                           dropout=0.1, gat_heads=4, gat_negative_slope=0.2, gat_bias=True)  # the drop-out is for both dropout in

        # self.graph_global_attention_pooling = MyConditionalGlobalAttention(
        #     num_node_features=self.scene_graph_encoder.sg_emb_dim,
        #     num_out_features=512)
        # self.tie_lm_weights()

    def forward(self,input_ids, sg_input=None, belief_input=None, labels=None,return_dict=False,output_attentions=None,
                    output_hidden_states=None, past_key_values=None, attention_mask=None, token_type_ids=None,
                            position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                            encoder_attention_mask=None, use_cache=None):
        #GAT
        #sg_input = torch_geometric.data.Batch.from_data_list(sg_input).to(input_ids.device)
        x_encoded, edge_attr_encoded,_ = self.scene_graph_encoder(sg_input)
        if self.with_ins:
            mem = self.transformer.transformer.wte(belief_input.T)
        else:
            mem = None
        x_executed = self.gat_seq(x=x_encoded, edge_index=sg_input.edge_index, edge_attr=edge_attr_encoded, instr_vectors=mem, batch=sg_input.batch)
        #print(("sg_input batch", sg_input.batch.shape, sg_input.batch.device, sg_input.ptr))
        #print(("inputs_ids", input_ids.shape, input_ids.device))
        # print(x_executed.shape, sg_input.batch.shape)
        dense_x_executed = torch_geometric.utils.to_dense_batch(x_executed,batch=sg_input.batch.clone())[0]
        #conv_labels_new = F.pad(labels, pad=(dense_x_executed.shape[1], 0), value=-100)

        #Concat GAT+GPT2 Input
        conv_input_embed = self.transformer.transformer.wte(input_ids)
        inputs_embeds = torch.cat((dense_x_executed, conv_input_embed ),dim =1)

        #GPT2
        dial_out = self.transformer(inputs_embeds=inputs_embeds, labels=labels,return_dict=return_dict,output_attentions=output_attentions,
                       output_hidden_states=output_hidden_states)
        # dial_out = self.transformer(input_ids=input_ids, labels=input_ids,return_dict=True,output_attentions=output_attentions,
        #                 output_hidden_states=output_hidden_states)
        # print(torch.argmax(torch.nn.functional.softmax(self.transformer(input_ids=input_ids[:, :50], return_dict=True, output_attentions=output_attentions,
        #                                                            output_hidden_states=output_hidden_states)['logits'][:, -1, :]), dim=1))
        return dial_out

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
            "belief_input": belief_input
        }
