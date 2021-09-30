import torch
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_mean, scatter_add
import ambigous.scripts.graph_representation.modified_graph_layernorm as m_layernorm
import torch_geometric

"""
Graph Meta Layer, Example funciton
"""
def __meta_layer():

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super(EdgeModel, self).__init__()
            self.edge_mlp = Sequential(Linear(2 * 10 + 5 + 20, 5), ReLU(), Linear(5, 5))

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super(NodeModel, self).__init__()
            self.node_mlp_1 = Sequential(Linear(15, 10), ReLU(), Linear(10, 10))
            self.node_mlp_2 = Sequential(Linear(2 * 10 + 20, 10), ReLU(), Linear(10, 10))

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out, u[batch]], dim=1)
            return self.node_mlp_2(out)

    class GlobalModel(torch.nn.Module):
        def __init__(self):
            super(GlobalModel, self).__init__()
            self.global_mlp = Sequential(Linear(20 + 10, 20), ReLU(), Linear(20, 20))

        def forward(self, x, edge_index, edge_attr, u, batch):
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
    return op


"""
Scene Graph Encoding Module For Ground Truth (Graph Neural Module)
Functional definition of scene graph encoding layer
Return: a callable operator, which is an initialized torch_geometric.nn graph neural layer
"""
def get_gt_scene_graph_encoding_layer(num_node_features, num_edge_features):

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super(EdgeModel, self).__init__()
            self.edge_mlp = Sequential(
                Linear(2 * num_node_features + num_edge_features, num_edge_features),
                ReLU(),
                Linear(num_edge_features, num_edge_features)
                )

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super(NodeModel, self).__init__()
            self.node_mlp_1 = Sequential(
                Linear(num_node_features + num_edge_features, num_node_features),
                ReLU(),
                Linear(num_node_features, num_node_features)
                )
            self.node_mlp_2 = Sequential(
                Linear(2 * num_node_features, num_node_features),
                ReLU(),
                Linear(num_node_features, num_node_features)
                )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(), NodeModel())
    return op

class sg_encoder(torch.nn.Module):
    def __init__(self, sg_vocab_size, sg_emb_dim,  pad_index,):
        super(sg_encoder, self).__init__()

        self.sg_emb_dim = sg_emb_dim # 300d glove
        self.sg_vocab_embedding = torch.nn.Embedding(sg_vocab_size, self.sg_emb_dim, padding_idx=pad_index)
        # self.sg_vocab_embedding.weight.data.copy_(sg_vocab.vectors)

        ##################################
        # build scene graph encoding layer
        ##################################
        self.scene_graph_encoding_layer = get_gt_scene_graph_encoding_layer(
            num_node_features=self.sg_emb_dim,
            num_edge_features=self.sg_emb_dim)

        self.graph_layer_norm = m_layernorm.LayerNorm(self.sg_emb_dim)

    def forward(self,gt_scene_graphs,):

        ##################################
        # Use glove embedding to embed ground truth scene graph
        ##################################
        # [ num_nodes, MAX_OBJ_TOKEN_LEN] -> [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim]
        x_embed     = self.sg_vocab_embedding(gt_scene_graphs.x)
        # [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim] -> [ num_nodes, sg_emb_dim]
        x_embed_sum = torch.sum(input=x_embed, dim=-2, keepdim=False)
        # [ num_edges, MAX_EDGE_TOKEN_LEN] -> [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim]
        edge_attr_embed = self.sg_vocab_embedding(gt_scene_graphs.edge_attr)

        # yanhao: for the manually added symmetric edges, reverse the sign of emb to denote reverse relationship:
        edge_attr_embed[gt_scene_graphs.added_sym_edge, :, :] *= -1


        # [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim] -> [ num_edges, sg_emb_dim]
        edge_attr_embed_sum   = torch.sum(input=edge_attr_embed, dim=-2, keepdim=False)
        del x_embed, edge_attr_embed

        ##################################
        # Call scene graph encoding layer
        ##################################
        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoding_layer(
            x=x_embed_sum,
            edge_index=gt_scene_graphs.edge_index,
            edge_attr=edge_attr_embed_sum,
            u=None,
            batch=gt_scene_graphs.batch
            )

        x_encoded = self.graph_layer_norm(x_encoded, gt_scene_graphs.batch)

        return x_encoded, edge_attr_encoded, None


