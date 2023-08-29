import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import dataset_spliter, get_configure, Evaluation
from argparse import ArgumentParser
from model.graph_predictor import Graph_predictor
from model.gnn import Pretrain_GNN
from dataset import DGL_dual_graph_set

def collate_molgraphs(data):
    ab_graphs, ba_graphs, labels, mask = map(list, zip(*data))

    abg = dgl.batch(ab_graphs)
    abg.set_n_initializer(dgl.init.zero_initializer)
    abg.set_e_initializer(dgl.init.zero_initializer)

    bag = dgl.batch(ba_graphs)
    bag.set_n_initializer(dgl.init.zero_initializer)
    bag.set_e_initializer(dgl.init.zero_initializer)

    labels = torch.stack(labels, dim=0)
    mask = torch.stack(mask, dim=0)

    return abg, bag, labels, mask

def train_epoch(data_loader, load_pretrain, model_list, loss_criterion, optimizer_list, args):
    graph_model = model_list[0]
    graph_model.train()
    if load_pretrain:
        pretrain_model = model_list[1]
        pretrain_model.train()
    evaluation = Evaluation(args['metric'])
    for batch_id, batch_data in enumerate(data_loader):
        abg, bag, labels, masks = batch_data
        abg = abg.to(args['device'])
        h = abg.ndata.pop('h')
        labels, masks = labels.to(args['device']), masks.to(args['device'])
        gh = None
        if load_pretrain:
            bag = bag.to(args['device'])
            e = bag.ndata.pop('e')
            n = bag.edata.pop('n')
            gh = pretrain_model(bag, e, n)
        prediction = graph_model(abg, h, gh)
        loss_mat = loss_criterion(prediction, labels) * masks
        loss = torch.sum(loss_mat)/torch.sum(masks)

        for optimizer in optimizer_list:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizer_list:
            optimizer.step()
        evaluation.update(prediction, labels, masks)
    train_score = evaluation.calculate()
    return train_score

def eval_epoch(data_loader, load_pretrain, model_list, args):
    graph_model = model_list[0]
    graph_model.eval()
    if load_pretrain:
        pretrain_model = model_list[1]
        pretrain_model.eval()
    evaluation = Evaluation(args['metric'])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            abg, bag, labels, masks = batch_data
            abg = abg.to(args['device'])
            h = abg.ndata.pop('h')
            labels, masks = labels.to(args['device']), masks.to(args['device'])
            gh = None
            if load_pretrain:
                bag = bag.to(args['device'])
                e = bag.ndata.pop('e')
                n = bag.edata.pop('n')
                gh = pretrain_model(bag, e, n)
            prediction = graph_model(abg, h, gh)
            evaluation.update(prediction, labels, masks)
        eval_score = evaluation.calculate()
    return eval_score

def main(config, args, train_set, val_set, test_set):
    train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set, batch_size=config['batch_size'], collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], collate_fn=collate_molgraphs)

    if args['pretrain']:
        config['pretrain'] = True
    else:
        config['pretrain'] = False
    model_list = []
    model = Graph_predictor(config).to(args['device'])
    model.reset_parameters()
    model_list.append(model)
    optimizer_list = []
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer_list.append(optimizer)

    #classification tasks
    if args['dataset'] in ['muv', 'bace', 'bbbp', 'clintox', 'sider', 'toxcast', 'hiv',  'tox21']:
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        args['metric'] = 'roc-auc'
        print('metric: roc-auc(higher is better)')
        best_score = -1

    #regression tasks
    if args['dataset'] in ['esol', 'freesolv', 'lipophilicity']:
        loss_criterion = nn.SmoothL1Loss(reduction='none')
        args['metric'] = 'rmse'
        print('metric: rmse(lower is better)')
        best_score = float("inf")

    if args['pretrain']:
        print('loading pretrained model...')
        pretrain_model = Pretrain_GNN(get_configure('pretrain')).to(args['device'])
        pretrain_model.reset_parameters()
        pretrain_model.load_state_dict(torch.load("model/saved/pretrained.pth"))
        model_list.append(pretrain_model)
        optimizer_list.append(Adam(pretrain_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']))

    if args['load_saved']:
        print('loading saved model...')
        args['num_epochs'] = 0
        checkpoint = torch.load('model/saved/{}.pth'.format(args['dataset']))
        model_list[0].load_state_dict(checkpoint['graph_predictor'])
        model_list[1].load_state_dict(checkpoint['finetuned_gnn'])

    early_stop = 0
    for epoch in range(args['num_epochs']):
        patience = config['patience']
        train_score = train_epoch(train_loader, args['pretrain'], model_list, loss_criterion, optimizer_list, args)
        val_score = eval_epoch(val_loader, args['pretrain'], model_list, args)
        if args['metric'] == 'roc-auc':
            if val_score >= best_score:
                best_score = val_score
                early_stop = 0
                if args['save_model'] is True:
                    torch.save({'graph_predictor': model_list[0].state_dict(),
                                'finetuned_gnn': model_list[1].state_dict()},
                               'model/saved/{}.pth'.format(args['dataset']))
            else:
                early_stop += 1
        if args['metric'] == 'rmse':
            if val_score <= best_score:
                best_score = val_score
                early_stop = 0
                if args['save_model'] is True:
                    torch.save({'graph_predictor': model_list[0].state_dict(),
                                'finetuned_gnn': model_list[1].state_dict()},
                               'model/saved/{}.pth'.format(args['dataset']))
            else:
                early_stop += 1
        test_score = eval_epoch(test_loader, args['pretrain'], model_list, args)
        print('epoch = {} , train_score: {:.4f}'.format(epoch + 1, train_score))
        print('early_stop = {}, val_score: {:.4f}, best_score: {:.4f}'.format(early_stop, val_score, best_score))
        print('test_score: {:.4f}'.format(test_score))
        if early_stop == patience:
            break

    if args['load_saved']:
        test_score = eval_epoch(test_loader, args['pretrain'], model_list, args)
        print('test_score: {:.4f}'.format(test_score))


if __name__ == '__main__':
    parser = ArgumentParser('Molecular property prediction')
    parser.add_argument('--dataset', choices=['bace', 'bbbp', 'clintox', 'sider',
                        'toxcast', 'hiv', 'tox21', 'muv', 'esol', 'freesolv', 'lipophilicity'], default='bbbp',
                        help='Dataset')
    parser.add_argument('--pretrain', default=True,
                        help='Whether to load the pre-trained model (default: True)')
    parser.add_argument('--load_saved', default=True,
                        help='Whether to load the saved model (default: False)')
    parser.add_argument('--save_model', default=False,
                        help='Whether to save the model (default: False)')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Maximum number of epochs for training(default: 300)')
    parser.add_argument('--split', choices=['scaffold', 'random'], default='scaffold', help="random or scaffold")

    args = parser.parse_args().__dict__
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    config = get_configure(args['dataset'])
    dataset = DGL_dual_graph_set(args['dataset'])
    train_set, val_set, test_set = dataset_spliter(args['split'], dataset, args['dataset'])
    main(config, args, train_set, val_set, test_set)







