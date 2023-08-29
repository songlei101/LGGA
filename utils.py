import json
import os
from collections import defaultdict
import numpy as np
import torch
from dgl.data.utils import Subset, split_dataset
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, FastFindRings
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def get_ordered_scaffold_sets(molecules, log_every_n, scaffold_func):
    assert scaffold_func in ['decompose', 'smiles'], \
        "Expect scaffold_func to be 'decompose' or 'smiles', " \
        "got '{}'".format(scaffold_func)
    if log_every_n is not None:
        print('Start computing Bemis-Murcko scaffolds.')
    scaffolds = defaultdict(list)
    for i, mol in enumerate(molecules):
        count_and_log('Computing Bemis-Murcko for compound',
                      i, len(molecules), log_every_n)
        try:
            FastFindRings(mol)
            if scaffold_func == 'decompose':
                mol_scaffold = Chem.MolToSmiles(AllChem.MurckoDecompose(mol))
            if scaffold_func == 'smiles':
                mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            scaffolds[mol_scaffold].append(i)
        except:
            print('Failed to compute the scaffold for molecule {:d} '
                  'and it will be excluded.'.format(i + 1))
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    return scaffold_sets
def train_val_test_sanity_check(frac_train, frac_val, frac_test):
    total_fraction = frac_train + frac_val + frac_test
    assert np.allclose(total_fraction, 1.), \
        'Expect the sum of fractions for training, validation and ' \
        'test to be 1, got {:.4f}'.format(total_fraction)

def count_and_log(message, i, total, log_every_n):
    if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i + 1, total))

def prepare_mols(dataset_name):
    log_every_n = 500
    df = pd.read_csv('data/raw/{}.csv'.format(dataset_name))
    mols = []
    if dataset_name in ['tox21', 'clintox', 'bbbp', 'sider', 'hiv',
                        'toxcast', 'muv', 'esol', 'freesolv', 'lipophilicity']:
        smiles_column = 'smiles'
    if dataset_name in ['bace']:
        smiles_column = 'mol'
    smiles = df[smiles_column].tolist()
    for i, s in enumerate(smiles):
        count_and_log('Creating RDKit molecule instance', i, len(smiles), log_every_n)
        mol = Chem.MolFromSmiles(s, sanitize=True)
        if mol is not None:
            mols.append(mol)
    return mols

def train_val_test_split(dataset, dataset_name, frac_train=0.8,
                         frac_val=0.1, frac_test=0.1,
                         log_every_n=1000, scaffold_func='decompose'):
    train_val_test_sanity_check(frac_train, frac_val, frac_test)
    molecules = prepare_mols(dataset_name)
    scaffold_sets = get_ordered_scaffold_sets(
        molecules, log_every_n, scaffold_func)
    train_indices, val_indices, test_indices = [], [], []
    train_cutoff = int(frac_train * len(molecules))
    val_cutoff = int((frac_train + frac_val) * len(molecules))
    for group_indices in scaffold_sets:
        if len(train_indices) + len(group_indices) > train_cutoff:
            if len(train_indices) + len(val_indices) + len(group_indices) > val_cutoff:
                test_indices.extend(group_indices)
            else:
                val_indices.extend(group_indices)
        else:
            train_indices.extend(group_indices)
    return [Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices)]

def dataset_spliter(type, dataset, dataset_name):

    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    if type == 'scaffold':
        train_set, val_set, test_set = train_val_test_split(
            dataset, dataset_name, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif type == 'random':
        train_set, val_set, test_set = split_dataset(dataset, frac_list=[train_ratio, val_ratio, test_ratio],
                                                     shuffle=True, random_state=None)
    return train_set, val_set, test_set

def get_configure(dataset):
    file_path = 'configures/{}.json'.format(dataset)
    if not os.path.isfile(file_path):
        return NotImplementedError('can not find the target')
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


class Evaluation():
    def __init__(self, metric):
        self.y_pred_list = []
        self.y_true_list = []
        self.mask = []
        self.metric = metric

    def update(self, y_pred, y_true, mask):
        self.y_pred_list.append(y_pred.detach().cpu())
        self.y_true_list.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def calculate(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred_list, dim=0)
        y_true = torch.cat(self.y_true_list, dim=0)

        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            m = mask[:, task]
            true = y_true[:, task][m != 0]
            pred = y_pred[:, task][m != 0]
            task_score = None
            if self.metric == 'roc-auc':
                try:
                    task_score = roc_auc_score(true.long().numpy(), torch.sigmoid(pred).numpy())
                except ValueError:
                    pass
            if self.metric == 'rmse':
                task_score = torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
            if task_score is not None:
                scores.append(task_score)
        return np.mean(scores)

