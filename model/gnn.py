import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.ops import edge_softmax
import torch

class LGGAT(nn.Module):
    def __init__(self, config):
        super(LGGAT, self).__init__()
        self.node_in_feats = config['node_in_feats']
        self.graph_hidden_feats = config['graph_hidden_feats']
        self.message_passing_num = config['num_step_message_passing']
        self.W_n = nn.Linear(self.node_in_feats, self.graph_hidden_feats)
        self.drop = nn.Dropout(0.1)
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.message_passing_num):
            self.gnn_layers.append(Graphconv(self.graph_hidden_feats, config['pretrain']))

        self.bias = nn.Parameter(torch.Tensor(self.graph_hidden_feats))


    def reset_parameters(self):
        self.W_n.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, gh):
        node_feats = self.W_n(node_feats)
        node_feats_list = []
        node_feats = self.drop(node_feats)
        node_feats_list.append(node_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, gh)
            node_feats_list.append(node_feats)
        return node_feats_list

class Graphconv(nn.Module):
    def __init__(self, graph_hidden_feats, load_pretrain):
        super(Graphconv, self).__init__()
        self.eps = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(graph_hidden_feats, graph_hidden_feats),
            nn.ReLU(),
            nn.Linear(graph_hidden_feats, 1)
        )
        self.load_pretrain = load_pretrain
        if self.load_pretrain:
            dim = 3
        else:
            dim = 2
        self.calculate_a = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(dim * graph_hidden_feats, graph_hidden_feats),
            nn.ReLU(),
            nn.Linear(graph_hidden_feats, 1)
        )

    def reset_parameters(self):
        for layer in self.eps:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.calculate_a:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def weight(self, edges):
        if self.load_pretrain:
            a = self.calculate_a(torch.cat([edges.src['h'], edges.dst['h'], edges.data['gh']], dim=-1))
        else:
            a = self.calculate_a(torch.cat([edges.src['h'], edges.dst['h']], dim=-1))
        return {'a': a}

    def message_func(self, edges):
        mul = torch.mul(edges.src['h'], edges.data['gh'])
        a = edges.data['a']
        return {'m': a*mul}

    def forward(self, g, node_feats, gh):
        g = g.local_var()
        g.ndata['h'] = node_feats
        if gh is None:
            g.apply_edges(self.weight)
            g.edata['a'] = edge_softmax(g, g.edata['a'])
            g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'hv'))
        else:
            g.edata['gh'] = gh
            g.apply_edges(self.weight)
            g.edata['a'] = edge_softmax(g, g.edata['a'])
            g.update_all(self.message_func, fn.sum('m', 'hv'))
        h = (1 + self.eps(node_feats)) * g.ndata['hv'] + node_feats
        return h


class Pretrain_GNN(nn.Module):
    def __init__(self, config):
        super(Pretrain_GNN, self).__init__()
        self.edge_in_feats = config['edge_in_feats']
        self.node_in_feats = config['node_in_feats']
        self.graph_hidden_feats = config['graph_hidden_feats']
        self.eps = nn.Parameter(torch.FloatTensor([0]))
        self.W_e = nn.Linear(self.edge_in_feats, self.graph_hidden_feats)
        self.W_n = nn.Linear(self.node_in_feats, self.graph_hidden_feats)
        self.fc = nn.Sequential(
            nn.Linear(self.graph_hidden_feats, self.graph_hidden_feats),
            nn.ReLU(),
            nn.Linear(self.graph_hidden_feats, self.graph_hidden_feats)
        )

    def reset_parameters(self):
        self.W_e.reset_parameters()
        self.W_n.reset_parameters()
        nn.init.zeros_(self.eps)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def src_dst(self, edges):
        angle_emb = torch.cat([edges.src['gh'], edges.dst['gh']], dim=-1)
        return {'angle_emb': angle_emb}

    def pretrain(self, g, e, n, d, angle, predictor_list, loss_criterion):
        e = self.W_e(e)
        n = self.W_n(n)
        g = g.local_var()
        g.ndata['e'] = e
        g.edata['n'] = n
        g.update_all(fn.u_mul_e('e', 'n', 'm'), fn.sum('m', 'gh'))
        g.ndata['gh'] = self.fc(g.ndata['gh']) + (1 + self.eps) * e
        d_pred = predictor_list[0](g.ndata['gh'])
        loss = loss_criterion(d_pred, d)
        g.apply_edges(self.src_dst)
        angle_pred = predictor_list[1](g.edata['angle_emb'])
        loss += loss_criterion(angle_pred, angle)
        return loss

    def forward(self, g, e, n):
        e = self.W_e(e)
        n = self.W_n(n)
        g = g.local_var()
        g.ndata['e'] = e
        g.edata['n'] = n
        g.update_all(fn.u_mul_e('e', 'n', 'm'), fn.sum('m', 'gh'))
        gh = self.fc(g.ndata['gh']) + (1 + self.eps) * e
        return gh

