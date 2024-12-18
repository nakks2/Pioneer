import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
from tqdm import tqdm
import math
from scipy.linalg import block_diag
import lib.utils as utils
import copy
import pandas as pd
import argparse
from scipy.sparse import coo_matrix
import os.path
import pickle

import networkx as nx
import random
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from collections import defaultdict


def compute_edge_weights_from_ndarray(y, graphs):
    """
    :param y: [T, N]
    :param graphs: [T, N, N]
    :return:[T, N, N]
    """
    T, N, _ = graphs.shape
    weighted_graphs = np.zeros((T, N, N))

    for t in range(T):
        for i in range(N):
            for j in range(N):
                if graphs[t, i, j] != 0:
                    weighted_graphs[t, i, j] = abs(y[t, i] - y[t, j])

    return weighted_graphs

def compute_edge_weights_from_coo(y, graphs):
    """
    :param y:[T, N]
    :param graphs: [T, N, N]
    :return:  [T, N, N]
    """
    T = graphs.shape[0]
    N = graphs[0].shape[0]
    weighted_graphs = []

    for t in range(T):
        coo = graphs[t]
        row_indices = coo.row
        col_indices = coo.col
        num_edges = len(row_indices)

        weights = np.zeros(num_edges)
        for i in range(num_edges):
            u = row_indices[i]
            v = col_indices[i]
            weights[i] = np.linalg.norm(y[t, u] - y[t, v])

        weighted_graph = coo_matrix((weights, (row_indices, col_indices)), shape=(N, N))
        weighted_graphs.append(weighted_graph)

    return np.array(weighted_graphs)

class ParseData(object):

    def __init__(self,args):
        self.args = args
        self.datapath = args.datapath
        self.dataset = args.dataset
        self.random_seed = args.random_seed
        self.pred_length = args.pred_length
        self.condition_length = args.condition_length
        self.batch_size = args.batch_size

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def feature_norm(self, features):
        one_feature = np.ones_like(features[:, 1:, :])  # [N,T-1,D]
        one_feature[:, :, :2] = features[:, 1:, :2] - features[:, :-1, :2]

        one_feature[:, :, -1] = features[:, 1:, -1]

        return one_feature

    def load_train_data(self,is_train = True):
        features, graphs = self.load_different_data()
        features = features[1:self.args.training_end_time+1, :,:]  # [T,N,D]
        graphs = graphs[:self.args.training_end_time - 1, :,:]  # [T,N,N]
        features = np.transpose(features, (1, 0, 2))
        if self.args.add_popularity and self.args.dataset=="social":
            features = self.add_popularity(features)
        features_original = copy.deepcopy(features[:, 1:, :])
        graphs_original = copy.deepcopy(graphs)
        features = self.feature_norm(features)
        self.num_states = features.shape[0]
        self.num_features = features.shape[2]
        # Split Training Samples
        features, graphs = self.generateTrainSamples(features, graphs)  # [K = 60,N,T,D], [K,T,N,N]
        features_original,_ = self.generateTrainSamples(features_original,graphs_original)

        k = 1
        if is_train:
            features = features[:-k, :, :, :]
            graphs = graphs[:-k, :, :, :]
            features_original = features_original[:-k,:,:,:]
        else:
            features = features[-k:, :, :, :]
            graphs = graphs[-k:, :, :, :]
            features_original = features_original[-k:, :, :, :]

        encoder_data_loader, decoder_data_loader,encoder_graph_loader,decoder_graph_loader ,num_batch = self.generate_train_val_dataloader(features,graphs,features_original)



        return encoder_data_loader, decoder_data_loader, encoder_graph_loader, decoder_graph_loader,num_batch,self.num_states,self.num_features

    def generate_train_val_dataloader(self, features, graphs,features_original):
        # Split data for encoder and decoder dataloader
        feature_observed, times_observed, series_decoder, times_extrap = self.split_data(features)  # series_decoder[K*N,T2,D]
        self.times_extrap = times_extrap

        #Generate gt
        _,_,series_decoder_gt,_ = self.split_data(features_original)


        # Generate Encoder data
        encoder_data_loader = self.transfer_data(feature_observed, graphs, times_observed, self.batch_size)

        # Generate Decoder Data and Graph

        series_decoder_all = [(series_decoder[i, :, :], series_decoder_gt[i, :, :]) for i in range(series_decoder.shape[0])]

        decoder_data_loader = Loader(series_decoder_all, batch_size=self.batch_size * self.num_states, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        graph_encoder = graphs[:, :self.args.condition_length, :, :]  # [K,T1,N,N]
        encoder_graph_loader = Loader(graph_encoder, batch_size=self.batch_size, shuffle=False)
        graph_decoder = graphs[:, self.args.condition_length:, :, :]
        decoder_graph_loader = Loader(graph_decoder, batch_size=self.batch_size, shuffle=False)
        num_batch = len(decoder_data_loader)
        assert len(encoder_data_loader) == len(encoder_graph_loader)

        # Inf-Generator
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        encoder_graph_loader = utils.inf_generator(encoder_graph_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)
        decoder_graph_loader = utils.inf_generator(decoder_graph_loader)
        return encoder_data_loader,decoder_data_loader,encoder_graph_loader,decoder_graph_loader,num_batch


    def load_test_data(self,pred_length,condition_length,ce=False):

        # Loading Data. N is state number, T is number of days. D is feature number.
        features, graphs = self.load_different_data()
        print("predicting data at: %s" % self.args.dataset)
        features = features[self.args.training_end_time - condition_length + 1-1:, :, :]  # [T=93,N,D]
        if ce:
            features = np.load(self.args.datapath + self.args.dataset + '/locations.npy')[0:41,:,:]
        features = np.transpose(features, (1, 0, 2))  # [N,T,D]
        graphs = graphs[self.args.training_end_time - condition_length:, :, :]  # [T,N,N]
        if ce:
            graphs = np.load(self.args.datapath + self.args.dataset+ '/graphs.npy')[0:41,:,:]
        self.num_states = features.shape[0]


        if self.args.add_popularity and self.args.dataset=="social":
            features = self.add_popularity(features)

        features_original = copy.deepcopy(features[:, 1:, :])
        graphs_original = copy.deepcopy(graphs)
        features = self.feature_norm(features)


        # Encoder
        features, graphs = self.generateTrainSamples(features, graphs)  # [K = 15,N,T,D], [K,T,N,N]

        features_enc = features[:, :, :condition_length, :]  # [K,N,T1,D]
        features_enc = features_enc
        graphs_enc = graphs[:,:condition_length,:,:]

        times_pred_max = pred_length
        times = np.asarray([i / (times_pred_max + condition_length) for i in
                            range(times_pred_max + condition_length)])  # normalized in [0,1] T
        times_observed = times[:condition_length]  # [T1]
        self.times_extrap = times[condition_length:] - times[condition_length]  # [T2] making starting time of T2 be 0.

        encoder_data_loader = self.transfer_data(features_enc, graphs_enc, times_observed,1)



        # Decoder data
        features_masks_dec = []  # K*[1,T,D]
        graphs_dec = []  # k*[1,T,N,N]
        features_original, _ = self.generateTrainSamples(features_original, graphs_original)

        for i, each_feature in enumerate(features):
            # decoder data
            features_each = each_feature[:,condition_length:,self.args.feature_out_index]  # [N,T2,D]
            tmp = features_original[i]
            features_each_origin = tmp[:, condition_length:, self.args.feature_out_index]  # [N,T2,D]

            graph_each = graphs[i,condition_length:, :, :] # [T2,N,N]
            graphs_dec.append(torch.FloatTensor(graph_each))  # K*[T=1,N,N]
            masks_each = np.asarray([i for i in range(pred_length)])
            features_masks_dec.append((features_each,features_each_origin, masks_each))

        decoder_graph_loader = Loader(graphs_dec, batch_size=1, shuffle=False)
        decoder_data_loader = Loader(features_masks_dec, batch_size=1, shuffle=False,
                                     collate_fn=lambda batch: self.variable_test(batch))

        # Inf-Generator
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        decoder_graph_loader = utils.inf_generator(decoder_graph_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        num_batch = features.shape[0]

        return encoder_data_loader, decoder_data_loader, decoder_graph_loader,num_batch



    def load_different_data(self):
        # Loading Data. N is state number, T is number of days. D is feature number.
        if self.dataset == 'social':

            features = np.load(self.args.datapath + self.args.dataset + '/locations.npy')
            graphs = np.load(self.args.datapath + self.args.dataset + '/graphs.npy')

            # Feature Preprocessing:
            self.features_max = features.max()  # 10.664
            self.features_min = features.min()  # -12.195
            #
            # # Normalize to [0,1]
            # features = (features - self.features_min) / (self.features_max - self.features_min)

        elif self.dataset  == "CReSIS":
            graphs, X, y = load_CReSIS(self.args.datapath, self.args.dataset, is_coo=True)
            self.features_max = y.max()  # 1788.22034
            self.features_min = y.min()  # 260.52
            y = (y - self.features_min) / (self.features_max - self.features_min)

            features = np.expand_dims(y, axis=2)
            graphs = self.transg(graphs)
            graphs = compute_edge_weights_from_ndarray(y, graphs)
        elif self.dataset  == "weather":
            graphs, X, y = load_weather(self.args.datapath,self.args.dataset)
            # Feature Preprocessing:
            self.features_max = y.max()  # 41.65714285714286
            self.features_min = y.min()  # -30.95714285714286

            # Normalize to [0,1]
            y = (y - self.features_min) / (self.features_max - self.features_min)

            graphs = compute_edge_weights_from_ndarray(y, graphs)
            features = np.expand_dims(y, axis=2)

        return features, graphs
    def generateTrainSamples(self,features, graphs):
        '''
        Split training data into several overlapping series.
        :param features: [N,T,D]/
        :param graphs: [T,N,N]
        :param interval: 3
        :return: transform feature into [K,N,T,D], transform graph into [K,T,N,N]
        '''
        interval = self.args.split_interval
        each_length = self.args.pred_length + self.args.condition_length
        num_batch = math.floor((features.shape[1] - each_length) / interval) + 1
        num_states = features.shape[0]
        num_features = features.shape[2]
        features_split = np.zeros((num_batch, num_states, each_length, num_features))
        graphs_split = np.zeros((num_batch, each_length, num_states, num_states))
        batch_num = 0

        for i in range(0, features.shape[1] - each_length+1, interval):
            assert i + each_length <= features.shape[1]
            features_split[batch_num] = features[:, i:i + each_length, :]
            graphs_split[batch_num] = graphs[i:i + each_length, :, :]
            batch_num += 1
        return features_split, graphs_split  # [K,N,T,D], [K,T,N,N]

    def split_data(self, feature):
        '''
               Generate encoder data (need further preprocess) and decoder data
               :param feature: [K,N,T,D], T=T1+T2
               :param data_type:
               :return:
               '''

        feature_observed = feature[:, :, :self.args.condition_length, :]
        # select corresponding features
        feature_out_index = self.args.feature_out_index
        feature_extrap = feature[:, :, self.args.condition_length:, feature_out_index]
        assert feature_extrap.shape[-1] == len(feature_out_index)
        times = np.asarray([i / feature.shape[2] for i in range(feature.shape[2])])  # normalized in [0,1] T
        times_observed = times[:self.args.condition_length]  # [T1]
        times_extrap = times[self.args.condition_length:] - times[
            self.args.condition_length]  # [T2] making starting time of T2 be 0.
        assert times_extrap[0] == 0
        series_decoder = np.reshape(feature_extrap, (-1, len(times_extrap), len(feature_out_index)))  # [K*N,T2,D]

        return feature_observed, times_observed, series_decoder, times_extrap

    def transfer_data(self, feature, edges, times,batch_size):
        '''

        :param feature: #[K,N,T1,D]
        :param edges: #[K,T,N,N], with self-loop
        :param times: #[T1]
        :param time_begin: 1
        :return:
        '''
        data_list = []
        edge_size_list = []

        num_samples = feature.shape[0]

        for i in tqdm(range(num_samples)):
            data_per_graph, edge_size = self.transfer_one_graph(feature[i], edges[i], times)
            data_list.append(data_per_graph)
            edge_size_list.append(edge_size)

        # print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=batch_size,shuffle=False)

        return data_loader



    def add_popularity(self, feature_input):
        '''
        Adding population data to features [N,T,D]
        :param feature_input: [N,T,D]
        :return: feature_output: [N,T,D+1]
        '''
        popularity = np.reshape(np.load(self.args.datapath + self.args.dataset + "/popularity.npy").astype("float"),(-1,1))  # [N,1]
        # normalize
        # popularity = (popularity - popularity.min())/(popularity.max() - popularity.min())
        popularity = np.expand_dims(popularity, axis=2)
        popularity_tensor = np.zeros((feature_input.shape[0], feature_input.shape[1], 1))
        popularity_tensor += popularity  # [N,T,1]
        feature_output = np.concatenate([feature_input, popularity_tensor], axis=2)

        return feature_output



    def transfer_one_graph(self,feature, edge, time):
        '''f

        :param feature: [N,T1,D]
        :param edge: [T,N,N]  (needs to transfer into [T1,N,N] first, already with self-loop)
        :param time: [T1]
        :param method:
            1. All -- preserve all cross-time edges
            2. Forward -- preserve cross-time edges where sender nodes are thosewhose time is smaller
            3. None -- no cross_time edges are preserved
        :param is_self_only:
            1. True: only preserve same-node cross-time edges
            2. False:
        :return:
            1. x : [N*T1,D]: feature for each node.
            2. edge_index [2,num_edge]: edges including cross-time
            3. edge_weight [num_edge]: edge weights
            4. y: [N], value= num_steps: number of timestamps for each state node.
            5. x_pos 【N*T1】: timestamp for each node
            6. edge_time [num_edge]: edge relative time.
        '''

        ########## Getting and setting hyperparameters:

        num_states = feature.shape[0]
        T1 = self.args.condition_length
        each_gap = 1/ edge.shape[0]
        edge = edge[:T1,:,:]
        time = np.reshape(time,(-1,1))

        ########## Compute Node related data:  x,y,x_pos
        # [Num_states],value is the number of timestamp for each state in the encoder, == args.condition_length
        y = self.args.condition_length*np.ones(num_states)
        # [Num_states*T1,D]
        x = np.reshape(feature,(-1,feature.shape[2]))
        # [Num_states*T1,1] node timestamp
        x_pos = np.concatenate([time for i in range(num_states)],axis=0)
        assert len(x_pos) == feature.shape[0]*feature.shape[1]

        ########## Compute edge related data
        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))], axis=0)  # [N*T1,N*T1], SAME TIME = 0

        edge_exist_matrix = np.ones((len(x_pos), len(x_pos)))

        # Step1: Construct edge_weight_matrix [N*T1,N*T1]
        edge_repeat = np.repeat(edge, self.args.condition_length, axis=2)  # [T1,N,NT1]
        edge_repeat = np.transpose(edge_repeat, (1, 0, 2))  # [N,T1,NT1]
        edge_weight_matrix = np.reshape(edge_repeat, (-1, edge_repeat.shape[2]))  # [N*T1,N*T1]

        # mask out cross_time edges of different state nodes.
        a = np.identity(T1)  # [T,T]
        b = np.concatenate([a for i in range(num_states)], axis=0)  # [N*T,T]
        c = np.concatenate([b for i in range(num_states)], axis=1)  # [N*T,N*T]

        a = np.ones((T1, T1))
        d = block_diag(*([a] * num_states))
        edge_weight_mask = (1 - d) * c + d
        edge_weight_matrix = edge_weight_matrix * edge_weight_mask  # [N*T1,N*T1]

        max_gap = each_gap


        # Step2: Construct edge_exist_matrix [N*T1,N*T1]: depending on both time and weight.
        edge_exist_matrix = np.where(
            (edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap) & (edge_weight_matrix != 0),
            edge_exist_matrix, 0)



        edge_weight_matrix = edge_weight_matrix * edge_exist_matrix
        edge_index, edge_weight_attr = utils.convert_sparse(edge_weight_matrix)
        assert np.sum(edge_weight_matrix!=0)!=0

        edge_time_matrix = (edge_time_matrix + 3) * edge_exist_matrix
        _, edge_time_attr = utils.convert_sparse(edge_time_matrix)
        edge_time_attr -= 3

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_weight_attr = torch.FloatTensor(edge_weight_attr)
        edge_time_attr = torch.FloatTensor(edge_time_attr)
        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)


        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_attr, y=y, pos=x_pos, edge_time = edge_time_attr)
        edge_num = edge_index.shape[1]

        return graph_data,edge_num


    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of
            - (feature0,feaure_gt) [K*N, T2, D]
        Returns:
            combined_tt: The union of all time observations. [T2]
            combined_vals: (M, T2, D) tensor containing the observed values.
        """
        # Extract corrsponding deaths or cases
        combined_vals = np.concatenate([np.expand_dims(ex[0],axis=0) for ex in batch],axis=0)
        combined_vals_true = np.concatenate([np.expand_dims(ex[1],axis=0) for ex in batch], axis = 0)



        combined_vals = torch.FloatTensor(combined_vals) #[M,T2,D]
        combined_vals_true = torch.FloatTensor(combined_vals_true)  # [M,T2,D]


        combined_tt = torch.FloatTensor(self.times_extrap)

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "data_gt" : combined_vals_true
            }
        return data_dict

    def variable_test(self,batch):
        """
        Expects a batch of
            - (feature,feature_gt,mask)
            - feature: [N,T,D]
            - mask: T
        Returns:
            combined_tt: The union of all time observations. [T2], varies from different testing sample
            combined_vals: (M, T2, D) tensor containing the observed values.
            combined_masks: index for output timestamps. Only for masking out prediction.
        """
        # Extract corrsponding deaths or cases

        combined_vals = torch.FloatTensor(batch[0][0]) #[M,T2,D]
        combined_vals_gt = torch.FloatTensor(batch[0][1]) #[M,T2,D]
        combined_masks = torch.LongTensor(batch[0][2]) #[1]

        combined_tt = torch.FloatTensor(self.times_extrap)

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "masks":combined_masks,
            "data_gt": combined_vals_gt,

            }
        return data_dict

    def transg(self,graphs):
        T = len(graphs)
        n = graphs[0].shape[0]
        g = np.zeros((T, n, n))
        for t in range(T):
            coo = graphs[t]
            g[t] = coo.toarray()
            # g[t] = g[t] + g[t].T
        return g

def load_CReSIS(path, name, is_coo, seed=15, num_points=100):
    if os.path.exists(path + name + "/y.npy"):
        X = np.load(path + name + "/X.npy")
        y = np.load(path + name + "/y.npy")
        graph = np.load(path + name + "/g.npy", allow_pickle=True)
        print("Load Success")
        # graph = graph + graph.T
        # graph = (graph > 0).astype(int)
        return graph, X, y

    np.random.seed(seed)
    random.seed(seed)

    point_values = defaultdict(lambda: {'X': [], 'y': []})
    point_count = defaultdict(int)

    for i in range(1993, 2014):
        print(f"Loading year: {i}")
        path_all = path + name + f"/processed_data/{i}/"
        try:
            X = np.load(path_all + "X.npy")
            y = np.load(path_all + "y.npy")
            coordinates = X[:, :2]

            coord_tuples = list(map(tuple, coordinates))
            for coord, x_row, y_val in zip(coord_tuples, X, y):
                point_values[coord]['X'].append(x_row)
                point_values[coord]['y'].append(y_val)
                point_count[coord] += 1
            negative_coords = np.argwhere(y < 0)

            for coord in negative_coords:
                print(f"Negative value {y[tuple(coord)]} found at coordinates {tuple(coord)}")
        except FileNotFoundError:
            continue

    sorted_points = sorted(point_count.items(), key=lambda item: item[1], reverse=True)
    selected_points = [point for point, _ in sorted_points[:num_points]]

    X_constant = np.array([np.mean(point_values[point]['X'], axis=0) for point in selected_points])

    point_stats = {}
    for point in selected_points:
        y_data = np.array(point_values[point]['y'])

        mean_y = np.mean(y_data)
        std_y = np.std(y_data)

        point_stats[point] = (mean_y, std_y)

    graphs, Xs, ys = [], [], []

    for i in range(1993, 2014):
        print(f"Processing year: {i}")
        path_all = path + name + f"/processed_data/{i}/"
        try:
            X = np.load(path_all + "X.npy")
            y = np.load(path_all + "y.npy")
            coordinates = X[:, :2]
            coord_tuples = list(map(tuple, coordinates))

            point_indices = {point: [] for point in selected_points}
            for idx, coord in enumerate(coord_tuples):
                if coord in point_indices:
                    point_indices[coord].append(idx)

            y_new = np.zeros((num_points,))
            mask = np.ones(num_points, dtype=bool)

            for idx, point in enumerate(selected_points):
                if point in point_indices:
                    idx_list = point_indices[point]
                    if idx_list:
                        y_new[idx] = np.mean(y[idx_list])
                        mask[idx] = False

            if np.any(mask):
                for idx in np.where(mask)[0]:
                    mean_y, std_y = point_stats[selected_points[idx]]
                    y_new[idx] = np.random.normal(mean_y, std_y)

            y_new = np.maximum(y_new, 0)

            Xs.append(X_constant)
            ys.append(y_new)

            graph = generate_graph(X_constant)
            if not is_coo:
                graph = graph.toarray()
            graphs.append(graph)

        except FileNotFoundError:
            continue

    np.save(path + name + "/X.npy", np.array(Xs))
    np.save(path + name + "/y.npy", np.array(ys))
    np.save(path + name + "/g.npy", np.array(graphs), allow_pickle=True)
    return np.array(graphs), np.array(Xs), np.array(ys)


def generate_graph(X, k=2):
    edges = set()
    df = pd.DataFrame(X, columns=['LON', 'LAT', 'TIME'])

    df['LON'] = df['LON'].round(k)
    df['LAT'] = df['LAT'].round(k)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            lon_min = min(df.loc[i, 'LON'], df.loc[j, 'LON'])
            lon_max = max(df.loc[i, 'LON'], df.loc[j, 'LON'])
            lat_min = min(df.loc[i, 'LAT'], df.loc[j, 'LAT'])
            lat_max = max(df.loc[i, 'LAT'], df.loc[j, 'LAT'])

            in_rectangle = df[
                (df['LON'] > lon_min) & (df['LON'] < lon_max) &
                (df['LAT'] > lat_min) & (df['LAT'] < lat_max)
                ]

            if in_rectangle.empty:
                edges.add((i, j))
                edges.add((j, i))

    if edges:
        edges = np.array(list(edges))
        row = edges[:, 0]
        col = edges[:, 1]
        data = np.ones(len(edges))
        graph = coo_matrix((data, (row, col)), shape=(len(X), len(X)))
    else:
        graph = coo_matrix((len(X), len(X)))

    return graph

def load_FF(path, name):
    # warning! X,Y/hours graph/years
    graphs, Xs, ys = [], [], []
    for i in range(2017, 2018):
        path_all = path + name + f"/yearly_data/{i}/"
        try:
            X = torch.load(path_all + f"node_attrs_{i}.pt").numpy()
            X = np.transpose(X, (1, 0, 2))
            y = X[:, :, :1]
            X = X[:, :, 1:]
            X[:, :, -1] /= 1000
            # y = np.load(path_all + "y.npy")
            edges = torch.load(path_all + f"edge_index_{i}.pt").numpy()
            graph = edges_to_adjacency_matrix(edges, len(X[0]))
            graphs.append(graph)
            Xs.extend(X)
            ys.extend(y)
        except FileNotFoundError:
            continue
    return np.array(graphs), np.array(Xs), np.array(ys).squeeze(-1)


def edges_to_adjacency_matrix(edges, num_nodes):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(edges.shape[1]):
        node1, node2 = edges[0][i], edges[1][i]
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Assuming the graph is undirected

    return adjacency_matrix


def load_weather(path, name):
    path = path + "weather/"
    with open(path + 'X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(path + 'y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open(path + 'g.pkl', 'rb') as f:
        graph = pickle.load(f)

    T = int(np.max(X[:, 1])) + 1
    N = 50
    D = 3

    X_new = np.zeros((T, N, D))

    y = np.array(y)

    average_edge_weights = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        weights = [data['weight'] for _, _, data in edges]
        if weights:
            average_edge_weights[node] = np.mean(weights)
        else:
            average_edge_weights[node] = 0.0

    for i in range(X.shape[0]):
        node = int(X[i, 0])
        time = int(X[i, 1])
        X_new[time, node, 0] = node
        X_new[time, node, 1] = time
        X_new[time, node, 2] = average_edge_weights[node]

    y_new = y.reshape(T, N)

    adj_matrix = nx.to_numpy_array(graph)

    graph_new = np.zeros((T, N, N))
    for t in range(T):
        graph_new[t] = adj_matrix

    return graph_new, X_new, y_new





