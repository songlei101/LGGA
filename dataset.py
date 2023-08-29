import os
from dgl.data.utils import load_graphs
from dual_graph_constructor import processing
from pretrain_graph_constructor import p_processing

class DGL_dual_graph_set():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.ab_graphs_path = 'data/processed/{}_ab_dglgraph.bin'.format(self.dataset_name)
        self.ba_graphs_path = 'data/processed/{}_ba_dglgraph.bin'.format(self.dataset_name)
        self._load()

    def _load(self):
        if os.path.exists(self.ab_graphs_path) and os.path.exists(self.ba_graphs_path):
            print('Loading previously saved dgl graphs...')
        else:
            processing(self.dataset_name)
        self.ab_graphs, dict1 = load_graphs(self.ab_graphs_path)
        self.ba_graphs, dict2 = load_graphs(self.ba_graphs_path)
        self.labels = dict1['labels']
        self.masks = dict1['mask']

    def __getitem__(self, item):
        return self.ab_graphs[item], self.ba_graphs[item], self.labels[item], self.masks[item]

    def __len__(self):
        return len(self.ab_graphs)


class DGL_pretrain_graph_set():
    def __init__(self):
        self.pretrain_graphs_path = 'data/processed/pretrain_graphs.bin'
        self._load()

    def _load(self):
        if os.path.exists(self.ab_graphs_path):
            print('Loading previously saved dgl graphs...')
        else:
            p_processing()
        self.ba_graphs, dict = load_graphs(self.pretrain_graphs_path)

    def __getitem__(self, item):
        return self.ba_graphs[item]

    def __len__(self):
        return len(self.ba_graphs)