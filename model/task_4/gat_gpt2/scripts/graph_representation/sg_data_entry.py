"""
This file provide data loading for phase 1 development.
We could ground truth scene graph from this file.

Need to refactor the graph loading components out.
"""
import json
import torch
import numpy as np
import torch_geometric


class sg_feature_lookup:


    def __init__(self, graph_path, tokenizer):
        self.graph = json.load(open(graph_path))
        self.tokenize = tokenizer
        # self.build_scene_graph_encoding_vocab(tokenizer)
        """
        Scene Graph Encoding:

        using package: https://pytorch-geometric.readthedocs.io/

        input: a scene graph in GQA format,
        output: a graph representation in pytorch geometric format - an instance of torch_geometric.data.Data


        Data Handling of Graphs

        A graph is used to model pairwise relations (edges) between objects (nodes).
        A single graph in PyTorch Geometric is described by an instance of torch_geometric.data.Data,
        which holds the following attributes by default:

        - data.x: Node feature matrix with shape [num_nodes, num_node_features]
        - data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        - data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        - [Not Applicable Here] data.y: Target to train against (may have arbitrary shape), e.g.,
                                        node-level targets of shape [num_nodes, *] or graph-level
                                        targets of shape [1, *]
        - [Not Applicable Here] data.pos: Node position matrix with shape [num_nodes, num_dimensions]

        """

    def convert_one_gqa_scene_graph(self, sg_this, tokenizer):

        ##################################
        # Make sure that it is not an empty graph
        ##################################
        scene = self.graph[sg_this]
        # SG_ENCODING_TEXT = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(scene.keys()) # str
        # map_objID_to_node_idx = {objID: node_idx for node_idx, objID in enumerate(objIDs)}
        nodes_list = set()



        nodes_list.update(objIDs)
        for key in objIDs:
            obj = scene[key]
            for attr in (obj['non-visual'] + obj['visual']):
                nodes_list.add(attr)
                # edge_list.add(attr.split('=')[0].strip())
            nodes_list.add('unique_id = ' + str(obj['unique_id']))

        map_nodes_to_node_idx = {nodeID: node_idx for node_idx, nodeID in enumerate(nodes_list)}

        node_feature_list = []
        edge_topology_list = []
        added_sym_edge_list = []
        edge_feature_list = []

        for key2 in objIDs:
            obj = scene[key2]
            obj_idx = map_nodes_to_node_idx[key2]
            for attr in (obj['non-visual'] + obj['visual']):
                attr_index = map_nodes_to_node_idx[attr]
                edge_topology_list.append([obj_idx, attr_index])
                edge_token_arr = np.array([tokenizer.convert_tokens_to_ids(attr.split('=')[0].strip())], dtype=np.int_) #tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                edge_feature_list.append(edge_token_arr)

                #Symmetric reverse connection
                edge_topology_list.append([attr_index, obj_idx])
                edge_token_arr_rev = np.array([tokenizer.convert_tokens_to_ids(attr.split('=')[0].strip())], dtype=np.int_) #tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                edge_feature_list.append(edge_token_arr_rev)
                added_sym_edge_list.append(len(edge_feature_list)-1)
            if key2 == 'Object ID: ROOT':
                # pass # Connect root to all
                for objID2 in objIDs:
                    obj_idx2 = map_nodes_to_node_idx[objID2]
                    edge_topology_list.append([obj_idx, obj_idx2])
                    edge_token_arr = np.array([tokenizer.convert_tokens_to_ids('FROM_ROOT')], dtype=np.int_)  # tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                    edge_feature_list.append(edge_token_arr)

                    #Symmetric from root
                    edge_topology_list.append([obj_idx2, obj_idx])
                    edge_token_arr = np.array([tokenizer.convert_tokens_to_ids('TO_ROOT')], dtype=np.int_)  # tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                    edge_feature_list.append(edge_token_arr)
            else:
                for rel in obj['relation']:
                    # [from self as source, to outgoing]
                    edge_topology_list.append([obj_idx, map_nodes_to_node_idx[rel[1]]])
                    # name of the relationship
                    edge_token_arr = np.array([tokenizer.convert_tokens_to_ids(rel[0])], dtype=np.int_) #tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                    edge_feature_list.append(edge_token_arr)

                    #REVERSE
                    edge_topology_list.append([map_nodes_to_node_idx[rel[1]],obj_idx])
                    edge_feature_list.append(edge_token_arr)
                    added_sym_edge_list.append(len(edge_feature_list)-1)



        for node in nodes_list:
            #Add features
            MAX_OBJ_TOKEN_LEN = 1 # only the name
            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int_) * self.tokenize.pad_token_type_id
            object_token_arr[0] = tokenizer.convert_tokens_to_ids(node)
            node_feature_list.append(object_token_arr)


            #Add self
            node_idx = map_nodes_to_node_idx[node]
            edge_topology_list.append([node_idx, node_idx])  # [from self, to self]
            edge_token_arr = np.array([tokenizer.convert_tokens_to_ids('<self>')], dtype=np.int_)  # tokenizer.vocab['<self>']
            edge_feature_list.append(edge_token_arr)


        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        datum = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        return datum

    def convert_one_gqa_scene_graph2(self, sg_this, tokenizer):

        ##################################
        # Make sure that it is not an empty graph
        ##################################
        scene = self.graph[sg_this]
        # SG_ENCODING_TEXT = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(scene.keys()) # str
        map_objID_to_node_idx = {objID: node_idx for node_idx, objID in enumerate(objIDs)}

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = [] # yanhao: record the index of added edges in the edge_feature_list

        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = scene[objId]
            for rel in obj['relation']:
                # [from self as source, to outgoing]
                from_to_connections_set.add((node_idx, map_objID_to_node_idx[rel[1]]))
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = scene[objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 19

            # 4 X '<pad>'
            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int_) * self.tokenize.pad_token_type_id

            # should have no error
            object_token_arr[0] = tokenizer.convert_tokens_to_ids(objId)  #mini_dict[objId]#tokenizer.vocab[objId]#SG_ENCODING_TEXT.vocab.stoi[objId]
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass
            # now use this constraint: 1–3 attributes
            # deduplicate

            ##################################
            # Comment out this to see the importance of attributes
            ##################################

            for attr_idx, attr in enumerate(set(scene[objId]['non-visual'] + scene[objId]['visual'])):
                object_token_arr[attr_idx + 1] = tokenizer.convert_tokens_to_ids(attr)#mini_dict[attr]#tokenizer.vocab[attr]

            node_feature_list.append(object_token_arr)

            ##################################
            # Need to Add a self-looping edge
            ##################################
            edge_topology_list.append([node_idx, node_idx]) # [from self, to self]
            edge_token_arr = np.array([tokenizer.convert_tokens_to_ids('<self>')], dtype=np.int_) #tokenizer.vocab['<self>']
            edge_feature_list.append(edge_token_arr)

            ##################################
            # Encode Edge
            # - Edge Feature: edge label (name)
            # - Edge Topology: adjacency matrix
            # GQA relations [dict]  A list of all outgoing relations (edges) from the object (source).
            ##################################


            ##################################
            # Comment out the whole for loop to see the importance of attributes
            ##################################

            for rel in obj['relation']:
                # [from self as source, to outgoing]
                edge_topology_list.append([node_idx, map_objID_to_node_idx[rel[1]]])
                # name of the relationship
                edge_token_arr = np.array([tokenizer.convert_tokens_to_ids(rel[0])], dtype=np.int_) #tokenizer.vocab[rel[0]]  #[SG_ENCODING_TEXT.vocab.stoi[rel[0]]]
                edge_feature_list.append(edge_token_arr)

                ##################################
                # Symmetric
                # - If there is no symmetric edge, add one.
                # - Should add mechanism to check duplicates
                ##################################
                if (map_objID_to_node_idx[rel[1]], node_idx) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append([map_objID_to_node_idx[rel[1]], node_idx])
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr)

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list)-1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        datum = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        return datum

