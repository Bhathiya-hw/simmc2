from copy import deepcopy

import torch

from torch.nn import Sequential, Linear, ReLU, Embedding, LayerNorm, Dropout, ModuleList
from torch_scatter import scatter_mean, scatter_add
import logging
import torch_geometric
import generation.scripts.graph_representation.sg_encoder as sg_encoder
import generation.scripts.graph_representation.gat_conv as gat_conv
logger = logging.getLogger(__name__)

"""
Transformer for text
"""
# helper class for the transformer decoder
import math

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


        # gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch.bmm(x.unsqueeze(1) , self.ques_nn(u)[batch].unsqueeze(2)).squeeze(-1)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self):
        return '{}(gate_nn={}, node_nn={}, ques_nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.node_nn, self.ques_nn)

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerProgramDecoder(torch.nn.Module):
    # should also be hierarchical

    def __init__(self, text_vocab_embedding, vocab_size, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerProgramDecoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        ##################################
        # For Hierarchical Deocding
        ##################################
        # TEXT = GQATorchDataset.TEXT
        self.num_queries = 1#GQATorchDataset.MAX_EXECUTION_STEP
        self.query_embed = torch.nn.Embedding(self.num_queries, ninp)

        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.coarse_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))

        ##################################
        # Decoding
        ##################################
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        self.ninp = ninp

        self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]

        ##################################
        # prepare target mask
        ##################################
        n_len_seq = tgt.shape[0] # seq len
        tgt_mask = self.generate_square_subsequent_mask(
                n_len_seq).to(memory.device)

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        tgt   = self.text_vocab_embedding(tgt)
        tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        ##################################
        # Replace the init token feature with instruciton feature
        ##################################

        tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
        tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

        output = self.transformer_decoder(tgt=tgt, memory=memory_repeat, tgt_mask=tgt_mask)
        output = self.vocab_decoder(output)

        # output both prediction and instruction vectors
        return output, instr_vectors

    def sample(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]


        tgt = None # discard

        max_output_len = 16 # 80 # program concat 80, full answer max 15, instr max 10
        batch_size = memory.size(1) * self.num_queries

        # TEXT = GQATorchDataset.TEXT
        output = torch.ones(max_output_len, batch_size).long().to(memory.device) * 0 #@todo


        for t in range(1, max_output_len):
            tgt = self.text_vocab_embedding(output[:t,:]) # from 0 to t-1
            tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
            tgt = self.pos_encoder(tgt) # contains dropout

            ##################################
            # Replace the init token feature with instruciton feature
            ##################################
            tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
            tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

            n_len_seq = t # seq len
            tgt_mask = self.generate_square_subsequent_mask(
                    n_len_seq).to(memory.device)
            # 2D mask (query L, key S)(L,S) where L is the target sequence length, S is the source sequence length.
            out = self.transformer_decoder(tgt, memory_repeat, tgt_mask=tgt_mask)
            # output: (T, N, E): target len, batch size, embedding size
            out = self.vocab_decoder(out)
            # target len, batch size, vocab size
            output_t = out[-1, :, :].data.topk(1)[1].squeeze()
            output[t,:] = output_t

        return output, instr_vectors

class TransformerFullAnswerDecoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, vocab_size, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerFullAnswerDecoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        self.ninp = ninp

        self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt):

        ##################################
        # prepare target mask
        ##################################
        n_len_seq = tgt.shape[0] # seq len
        tgt_mask = self.generate_square_subsequent_mask(
                n_len_seq).to(memory.device)

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        # print("tgt", tgt.size(),tgt)
        tgt   = self.text_vocab_embedding(tgt)
        # print("tgt", tgt.size())
        tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
        # print("tgt", tgt.size())
        tgt = self.pos_encoder(tgt)
        # print("tgt", tgt.size())
        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        output = self.vocab_decoder(output)

        return output

class TransformerQuestionEncoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerQuestionEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp) )
        self.ninp = ninp

    def forward(self, src):

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src   = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class GATEncoderDecoder(torch.nn.Module):
    def __init__(self, tokenizer, gat_conv_layers=5, ins_dim = 768, question_hidden_dim = 768):
        super(GATEncoderDecoder, self).__init__()
        self.ins_dim = ins_dim #self.transformer.transformer.wte.embedding_dim
        self.tokenizer = tokenizer
        self.scene_graph_encoder = sg_encoder.sg_encoder(len(self.tokenizer), self.ins_dim, self.tokenizer.pad_token_id)
        self.text_vocab_embedding = torch.nn.Embedding(len(self.tokenizer), self.ins_dim, padding_idx=self.tokenizer.pad_token_id)
        self.question_hidden_dim = question_hidden_dim  # 256, 79% slower # 128 - 82% on short # 512, batch size
        # self.text_vocab_embedding = torch.nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=text_pad_idx)
        self.question_encoder = TransformerQuestionEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=self.ins_dim,  # embedding dimension
            ninp=self.question_hidden_dim,  # transformer encoder layer input dim
            nhead=8,  # the number of heads in the multiheadattention models
            nhid=4 * self.question_hidden_dim,  # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1,  # the dropout value
        )

        ##################################
        # Build Program Decoder
        ##################################
        self.program_decoder = TransformerProgramDecoder(
            text_vocab_embedding=self.text_vocab_embedding,
            vocab_size=len(tokenizer),
            text_emb_dim=self.ins_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
            )

        self.gat_seq = gat_conv.gat_seq(in_channels=self.scene_graph_encoder.sg_emb_dim,
                           out_channels=self.scene_graph_encoder.sg_emb_dim,
                           edge_attr_dim=self.scene_graph_encoder.sg_emb_dim,
                           with_ins= False,
                           ins_dim=self.ins_dim, gat_conv_layers=gat_conv_layers,
                           dropout=0.1, gat_heads=4, gat_negative_slope=0.2, gat_bias=True)  # the drop-out is for both dropout in

        self.full_answer_decoder = TransformerFullAnswerDecoder(
            text_vocab_embedding=self.text_vocab_embedding,
            vocab_size=len(tokenizer),
            text_emb_dim=self.ins_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=8, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=3, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=0.1, # the dropout value
            )

        num_short_answer_choices = len(self.tokenizer)
        hid_dim = self.question_hidden_dim  * 3 # due to concat
        # self.logit_fc = torch.nn.Linear(hid_dim, num_short_answer_choices)
        # out_classifier_dim = 512
        # self.logit_fc = torch.nn.Sequential(
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(hid_dim, out_classifier_dim),
        #     torch.nn.ELU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(out_classifier_dim, num_short_answer_choices)
        # )

        self.graph_global_attention_pooling = MyConditionalGlobalAttention(
            num_node_features=self.scene_graph_encoder.sg_emb_dim,
            num_out_features=self.question_hidden_dim)

        # self.short_answer_criterion = torch.nn.CrossEntropyLoss()

        # self.graph_global_attention_pooling = MyConditionalGlobalAttention(
        #     num_node_features=self.scene_graph_encoder.sg_emb_dim,
        #     num_out_features=512)
        # self.tie_lm_weights()

    def forward(self,
                    questions,
                    gt_scene_graphs,
                    programs_input,
                    short_answers,
                    full_answers):
        #GAT
        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoder(gt_scene_graphs)

        ##################################
        # Encode questions
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        questions_encoded = self.question_encoder(questions)

        ##################################
        # Decode programs
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        # if not SAMPLE_FLAG:
        # programs_output, instr_vectors = self.program_decoder(memory=questions_encoded, tgt=programs_input)
        # else:
        #     programs_output, instr_vectors = self.program_decoder.sample(memory=questions_encoded, tgt=programs_input)

        ##################################
        # Call Recurrent Neural Execution Module
        ##################################
        # x_executed, execution_bitmap, history_vectors = self.recurrent_execution_engine(
        #     x=x_encoded,
        #     edge_index=gt_scene_graphs.edge_index,
        #     edge_attr=None,
        #     instr_vectors=instr_vectors,
        #     batch=gt_scene_graphs.batch,
        # )

        # print("inst: shape", instr_vectors.shape)
        # ins = instr_vectors[0] # shape: batch_size X instruction_dim
        # edge_batch = gt_scene_graphs.batch[gt_scene_graphs.edge_index[0]] # find out which batch the edge belongs to
        # repeated_ins = torch.zeros((gt_scene_graphs.edge_index.shape[1], ins.shape[-1])) # shape: num_edges x instruction_dim
        # repeated_ins = ins[edge_batch] # pick correct batched instruction for each edge

        # edge_cat = torch.cat( (edge_attr_encoded, repeated_ins.to(edge_attr_encoded.device)), dim=-1) # shape: num_edges X  encode_dim+instruction_dim
        # x_cat = torch.cat( (x_encoded, x_encoded), dim=-1)

        # x_executed = self.gat_seq(x=x_cat, edge_index=gt_scene_graphs.edge_index, edge_attr=edge_cat)

        x_executed = self.gat_seq(x=x_encoded, edge_index=gt_scene_graphs.edge_index, edge_attr=edge_attr_encoded, instr_vectors=None, batch=gt_scene_graphs.batch)

        ##################################
        # Final Layer of the Neural Execution Module, global pooling
        # (batch_size, channels)
        ##################################
        global_language_feature = questions_encoded[0]  # should be changed when completing NEM
        graph_final_feature = self.graph_global_attention_pooling(
            x=x_executed,  # x=x_encoded,
            u=global_language_feature,
            batch=gt_scene_graphs.batch,
            # no need for edge features since it is global node pooling
            size=None)

        ##################################
        # Call Short Answer Classification Module Only for Debug
        ##################################
        # short_answer_feature = questions_encoded[0]
        # short_answer_feature = torch.cat((graph_final_feature, questions_encoded[0], graph_final_feature * questions_encoded[0]), dim=-1)

        # short_answer_logits = self.logit_fc(short_answer_feature)
        # short_answer_logits =
        # print((torch.argmax(torch.nn.functional.softmax(short_answer_logits),dim=1), short_answers))
        # short_answer_loss = self.short_answer_criterion( short_answer_logits, short_answers.squeeze(0))


        return graph_final_feature #short_answer_loss # programs_output, short_answer_logits


