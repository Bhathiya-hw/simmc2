import torch

from torch.nn import Sequential, Linear, ReLU, Embedding, LayerNorm, Dropout, ModuleList
from torch_scatter import scatter_mean, scatter_add
from gat_gpt2.scripts.graph_representation.modified_graph_layernorm import LayerNorm as modifiedLayerNorm
import logging
import torch_geometric
from gat_gpt2.scripts.graph_representation.sg_encoder import sg_encoder
from gat_gpt2.scripts.graph_representation.gat_conv import gat, gat_seq

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
)

from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary

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


class Graph2Dial(PreTrainedModel):
    def __init__(self, config,tokenizer, pretrained_model_path=None, cache_dir= None ,add_special_tokens =True, with_ins =False, gat_conv_layers=5 ):
        super(Graph2Dial, self).__init__(config)
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
        self.scene_graph_encoder = sg_encoder(len(tokenizer), self.ins_dim, tokenizer.pad_token_id)

        self.gat_seq = gat_seq(in_channels=self.scene_graph_encoder.sg_emb_dim,
                           out_channels=self.scene_graph_encoder.sg_emb_dim,
                           edge_attr_dim=self.scene_graph_encoder.sg_emb_dim,
                           with_ins= self.with_ins,
                           ins_dim=self.ins_dim, gat_conv_layers=gat_conv_layers,
                           dropout=0.1, gat_heads=4, gat_negative_slope=0.2, gat_bias=True)  # the drop-out is for both dropout in

        self.graph_global_attention_pooling = MyConditionalGlobalAttention(
            num_node_features=self.scene_graph_encoder.sg_emb_dim,
            num_out_features=512)
        # self.tie_lm_weights()

    def forward(self,input_ids, sg_input=None, belief_input=None, labels=None,return_dict=False,output_attentions=None,
                    output_hidden_states=None, past_key_values=None, attention_mask=None, token_type_ids=None,
                            position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                            encoder_attention_mask=None, use_cache=None):
        #GAT
        x_encoded, edge_attr_encoded,_ = self.scene_graph_encoder(sg_input)
        if self.with_ins:
            mem = self.transformer.transformer.wte(belief_input.T)
        else:
            mem = None
        x_executed = self.gat_seq(x=x_encoded, edge_index=sg_input.edge_index, edge_attr=edge_attr_encoded, instr_vectors=mem, batch=sg_input.batch)
        # print(sg_input.batch.shape)
        dense_x_executed = torch_geometric.utils.to_dense_batch(x_executed,batch=sg_input.batch.clone())[0]

        #Concat GAT+GPT2 Input
        conv_input_embed = self.transformer.transformer.wte(input_ids)
        inputs_embeds = torch.cat((dense_x_executed, conv_input_embed ),dim =1)

        #GPT2
        dial_out = self.transformer(inputs_embeds=inputs_embeds, labels=labels,return_dict=return_dict,output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states)
        return dial_out

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

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