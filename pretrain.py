import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.gnn import Pretrain_GNN
from utils import get_configure
from dataset import DGL_pretrain_graph_set

def collate_molgraphs(data):
    graphs = data
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg

def train_epoch(data_loader, model, predictor_list, loss_criterion, optimizer_list):
    losses = 0.0
    model.train()
    predictor_list[0].train()
    predictor_list[1].train()
    for batch_id, batch_data in enumerate(data_loader):
        bg = batch_data
        bg = bg.to('cuda:0')
        e = bg.ndata.pop('e')
        d = bg.ndata.pop('d')
        angle = bg.edata.pop('angle')
        n = bg.edata.pop('n')
        loss = model.pretrain(bg, e, n, d, angle, predictor_list, loss_criterion)
        optimizer_list[0].zero_grad()
        optimizer_list[1].zero_grad()
        optimizer_list[2].zero_grad()
        loss.backward()
        optimizer_list[0].step()
        optimizer_list[1].step()
        optimizer_list[2].step()
        losses += loss.item()
    return losses

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    config = get_configure('pretrain')
    dataset = DGL_pretrain_graph_set()
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True,
                            collate_fn=collate_molgraphs)

    model = Pretrain_GNN(config).to(device)
    model.reset_parameters()

    predictor_list = []
    predictor_list.append(nn.Linear(config['graph_hidden_feats'], 1).to(device))
    predictor_list.append(nn.Linear(2 * config['graph_hidden_feats'], 1).to(device))

    loss_criterion = nn.SmoothL1Loss()

    optimizer_list = []
    optimizer_list.append(Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']))
    optimizer_list.append(Adam(predictor_list[0].parameters(), lr=config['lr'], weight_decay=config['weight_decay']))
    optimizer_list.append(Adam(predictor_list[1].parameters(), lr=config['lr'], weight_decay=config['weight_decay']))

    lowest_loss = float("inf")
    early_stop = 0
    patience = config['patience']

    for epoch in range(100):
        loss = train_epoch(dataloader, model, predictor_list, loss_criterion, optimizer_list)
        if loss < lowest_loss:
            early_stop = 0
            lowest_loss = loss
        else:
            early_stop += 1
        print('epoch:{} early_stop = {}, loss: {:.4f}'.format(epoch + 1, early_stop, loss))
        if early_stop == patience:
            break
    torch.save(model.state_dict(), "model/saved/pretrained.pth")



