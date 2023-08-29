import dgl
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from dgl import backend as F
from dgl.data.utils import save_graphs
from rdkit import Chem
from rdkit.Chem import AllChem

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_set = [
        'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
        'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'Others']
    h = []
    h += one_of_k_encoding_unk(atom.GetSymbol(), possible_set)  # 16
    h += one_of_k_encoding_unk(atom.GetDegree(), list(range(6)))  # 6
    h.append(atom.GetFormalCharge())  # 1
    h.append(atom.GetNumRadicalElectrons())  # 1
    h += one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                         Chem.rdchem.HybridizationType.SP2,
                                                         Chem.rdchem.HybridizationType.SP3,
                                                         Chem.rdchem.HybridizationType.SP3D,
                                                         Chem.rdchem.HybridizationType.SP3D2,
                                                         'Others'
                                                         ])  # 6
    h.append(atom.GetIsAromatic())  # 1
    h += one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(5)))  # 5
    h.append(atom.HasProp('_ChiralityPossible'))  # 1
    if not atom.HasProp('_CIPCode'):
        h += [False, False]
    else:
        h += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])  # 2
    return h

def get_bond_features(bond):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    possible_set = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'Others']
    h = []
    atom1_f = one_of_k_encoding_unk(atom1.GetSymbol(), possible_set)
    atom2_f = one_of_k_encoding_unk(atom2.GetSymbol(), possible_set)
    atom1_f.extend([atom1.GetFormalCharge(), atom1.GetNumRadicalElectrons(), atom1.GetTotalNumHs()])
    atom2_f.extend([atom2.GetFormalCharge(), atom2.GetNumRadicalElectrons(), atom2.GetTotalNumHs()])
    h += list(map(lambda x, y: x + y, atom1_f, atom2_f))  # 19

    bond_type = bond.GetBondType()
    h += one_of_k_encoding_unk(bond_type, [Chem.rdchem.BondType.SINGLE,
                                           Chem.rdchem.BondType.DOUBLE,
                                           Chem.rdchem.BondType.TRIPLE,
                                           Chem.rdchem.BondType.AROMATIC])
    h.append(bond.GetIsConjugated())
    h.append(bond.IsInRing())
    bond_stereo = bond.GetStereo()
    h += one_of_k_encoding_unk(bond_stereo, [Chem.rdchem.BondStereo.STEREONONE,
                                             Chem.rdchem.BondStereo.STEREOANY,
                                             Chem.rdchem.BondStereo.STEREOZ,
                                             Chem.rdchem.BondStereo.STEREOE])
    return h

def abg_node_featurizer(mol):
    atom_feats_dict = defaultdict(list)
    num_atoms = mol.GetNumAtoms()
    for node_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(node_idx)
        h = get_atom_features(atom)
        atom_feats_dict['h'].append(F.tensor(np.array(h).astype(np.float32)))
    atom_feats_dict['h'] = F.stack(atom_feats_dict['h'], dim=0)
    return atom_feats_dict

def bag_node_featurizer(mol):
    node_feats_dict = defaultdict(list)
    num_nodes = mol.GetNumBonds()
    for bond_idx in range(num_nodes):
        bond = mol.GetBondWithIdx(bond_idx)
        h = get_bond_features(bond)
        node_feats_dict['e'].append(F.tensor(np.array(h).astype(np.float32)))
    node_feats_dict['e'] = F.stack(node_feats_dict['e'], dim=0)
    node_feats_dict['e'] = torch.cat([node_feats_dict['e'], node_feats_dict['e']], dim=0)
    return node_feats_dict

def construct_graphs(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    bag = dgl.graph(([], []), idtype=torch.int32)
    abg = dgl.graph(([], []), idtype=torch.int32)
    num_atoms = mol.GetNumAtoms()
    num_edges = mol.GetNumBonds()

    abg.add_nodes(num_atoms)
    n_src = []
    n_dst = []
    for e_idx in range(num_edges):
        bond = mol.GetBondWithIdx(e_idx)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        n_src.append(u)
        n_dst.append(v)
    t = n_src.copy()
    n_src += n_dst
    n_dst += t

    abg.add_edges(torch.IntTensor(n_src), torch.IntTensor(n_dst))
    abg.ndata.update(abg_node_featurizer(mol))

    if num_edges>0:
        bag.add_nodes(2*num_edges)
        e_dict = {}
        for x in range(2*num_edges):
            e_dict[(n_src[x], n_dst[x])] = x
        adj = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    if mol.GetBondBetweenAtoms(i, j) is None: continue
                    adj[i, j] = 1
        e_src = []
        e_dst = []
        h_n = []
        for atom_idx in range(num_atoms):
            l = []
            for adj_atom, i in enumerate(adj[atom_idx]):
                if i == 1:
                    l.append(adj_atom)
            if len(l) > 1:
                for i in range(len(l) - 1):
                    atom1 = l[0]
                    l.remove(atom1)
                    for atom2 in l:
                        idx1 = e_dict[(atom_idx, atom1)]
                        idx2 = e_dict[(atom_idx, atom2)]
                        e_src.extend([idx1, idx2])
                        e_dst.extend([idx2, idx1])
                        f = get_atom_features(mol.GetAtomWithIdx(atom_idx))
                        h_n.extend(f)
                        h_n.extend(f)
        bag.add_edges(torch.IntTensor(e_src), torch.IntTensor(e_dst))
        bag.ndata.update(bag_node_featurizer(mol))
        bag.edata['n'] = F.tensor(np.array(h_n).astype(np.float32)).reshape(-1, 39)
    return abg, bag


def processing(dataset_name):
    drop_list = []
    task_names = None
    smiles_column = None

    if dataset_name == 'esol':
        task_names = ['measured log solubility in mols per litre']
        smiles_column = 'smiles'

    if dataset_name == 'freesolv':
        task_names = ['expt']
        smiles_column = 'smiles'

    if dataset_name == 'lipophilicity':
        task_names = ['exp']
        smiles_column = 'smiles'

    if dataset_name == 'tox21':
        smiles_column = 'smiles'
        drop_list.append(smiles_column)
        drop_list.append('mol_id')

    if dataset_name == 'muv':
        smiles_column = 'smiles'
        drop_list.append(smiles_column)
        drop_list.append('mol_id')

    if dataset_name == 'clintox':
        smiles_column = 'smiles'
        drop_list.append(smiles_column)

    if dataset_name == 'toxcast':
        smiles_column = 'smiles'
        drop_list.append(smiles_column)

    if dataset_name == 'bace':
        task_names = ['Class']
        smiles_column = 'mol'

    if dataset_name == 'bbbp':
        task_names = ['p_np']
        smiles_column = 'smiles'

    if dataset_name == 'sider':
        smiles_column = 'smiles'
        drop_list.append(smiles_column)

    if dataset_name == 'hiv':
        task_names = ['HIV_active']
        smiles_column = 'smiles'

    df = pd.read_csv('data/raw/{}.csv'.format(dataset_name))
    smiles = df[smiles_column].tolist()
    ab_graphs = []
    ba_graphs = []
    log_every = 500
    for i, s in enumerate(smiles):
        if (i + 1) % log_every == 0:
            print('Processing molecule {:d}/{:d}'.format(i + 1, len(smiles)))
        abg, bag = construct_graphs(s)
        ab_graphs.append(abg)
        ba_graphs.append(bag)
    # Keep only valid molecules
    valid_idx = []
    ab_valid_graphs = []
    ba_valid_graphs = []
    for idx in range(len(ab_graphs)):
        if ab_graphs[idx] is not None:
            valid_idx.append(idx)
            ab_valid_graphs.append(ab_graphs[idx])
            ba_valid_graphs.append(ba_graphs[idx])
    if task_names is None:
        for column in drop_list:
            df = df.drop(columns=[column])
        task_names = df.columns.tolist()
    print('{} molecules has been processed'.format(len(ab_valid_graphs)))
    labels_value = df[task_names].values
    labels = F.zerocopy_from_numpy(np.nan_to_num(labels_value).astype(np.float32))[valid_idx]
    mask = F.zerocopy_from_numpy((~np.isnan(labels_value)).astype(np.float32))[valid_idx]
    save_graphs('data/processed/{}_ab_dglgraph.bin'.format(dataset_name), ab_valid_graphs,
                labels={'labels': labels, 'mask': mask})
    save_graphs('data/processed/{}_ba_dglgraph.bin'.format(dataset_name), ba_valid_graphs)

