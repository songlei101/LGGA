import dgl
import numpy as np
import torch
from collections import defaultdict
from dgl import backend as F
from dgl.data.utils import save_graphs
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

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

def bag_node_featurizer(mol):
    #get mol conformer
    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    conf = mol.GetConformer(0)

    node_feats_dict = defaultdict(list)
    num_nodes = mol.GetNumBonds()
    for bond_idx in range(num_nodes):
        bond = mol.GetBondWithIdx(bond_idx)
        node_feats_dict['d'].append(rdMolTransforms.GetBondLength(conf,
                                                                  bond.GetBeginAtomIdx(),
                                                                  bond.GetEndAtomIdx()))
        h = get_bond_features(bond)
        node_feats_dict['e'].append(F.tensor(np.array(h).astype(np.float32)))
    node_feats_dict['e'] = F.stack(node_feats_dict['e'], dim=0)
    node_feats_dict['e'] = torch.cat([node_feats_dict['e'], node_feats_dict['e']], dim=0)
    node_feats_dict['d'] = F.tensor(np.array(node_feats_dict['d']).astype(np.float32)).reshape(-1, 1)
    node_feats_dict['d'] = torch.cat([node_feats_dict['d'], node_feats_dict['d']], dim=0)
    return node_feats_dict

def contruct_ba_graphs(mol):
    bag = dgl.graph(([], []), idtype=torch.int32)
    conf = mol.GetConformer(0)
    num_atoms = mol.GetNumAtoms()
    num_edges = mol.GetNumBonds()
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
    e_dict = {}
    for x in range(2 * num_edges):
        e_dict[(n_src[x], n_dst[x])] = x

    bag.add_nodes(2 * num_edges)
    adj = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                if mol.GetBondBetweenAtoms(i, j) is None: continue
                adj[i, j] = 1
    e_src = []
    e_dst = []
    h_n = []
    angle_list = []
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
                    angle = rdMolTransforms.GetAngleRad(conf, atom1, atom_idx, atom2)
                    idx1 = e_dict[(atom_idx, atom1)]
                    idx2 = e_dict[(atom_idx, atom2)]
                    e_src.extend([idx1, idx2])
                    e_dst.extend([idx2, idx1])
                    f = get_atom_features(mol.GetAtomWithIdx(atom_idx))
                    h_n.extend(f)
                    h_n.extend(f)
                    angle_list.extend([angle, angle])
    bag.add_edges(torch.IntTensor(e_src), torch.IntTensor(e_dst))
    bag.ndata.update(bag_node_featurizer(mol))
    bag.edata['angle'] = F.tensor(np.array(angle_list).astype(np.float32)).reshape(-1, 1)
    bag.edata['n'] = F.tensor(np.array(h_n).astype(np.float32)).reshape(-1, 39)
    return bag

def p_processing():
    file_path = r'data/raw/pretrain_mols.sdf'
    supp = Chem.SDMolSupplier(file_path)
    ba_graphs = []
    count = 0
    for mol in supp:
        count += 1
        if count % 1000 == 0:
            print('Processing molecule {:d}'.format(count))
        ba_graph = contruct_ba_graphs(mol)
        ba_graphs.append(ba_graph)
    save_graphs('data/processed/pretrain_graphs.bin', ba_graphs)



