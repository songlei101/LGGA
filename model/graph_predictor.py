import torch.nn as nn
from .graph_pooling import Set2Set
from .gnn import LGGAT

class Graph_predictor(nn.Module):
    def __init__(self, config):
        super(Graph_predictor, self).__init__()
        self.graph_hidden_feats = config['graph_hidden_feats']
        self.node_out_feats = config['node_out_feats']
        self.graph_feats = config['graph_feats']
        self.graph_layer = LGGAT(config)
        self.readout = Set2Set(input_dim=self.node_out_feats,
                               n_iters=config['num_step_set2set'],
                               n_layers=config['num_layer_set2set'])
        self.predict = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.graph_feats, self.node_out_feats),
            nn.ReLU(),
            nn.Linear(self.node_out_feats, config['n_tasks'])
        )

    def reset_parameters(self):
        self.graph_layer.reset_parameters()
        self.readout.reset_parameters()
        for layer in self.predict:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, g, node_feats, gh):
        node_feats_list = self.graph_layer(g, node_feats, gh)
        graph_feats = self.readout(g, node_feats_list[-1])
        return self.predict(graph_feats)