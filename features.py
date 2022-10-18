import rdkit
import numpy as np
import torch
import networkx
from rdkit.Chem import AllChem

DIM_ATOM_CATE_FEAT = 16
DIM_ATOM_FLOAT_FEAT = 1
DIM_BOND_CATE_FEAT = 6
DIM_BOND_FLOAT_FEAT = 2

DICT_CHIRALTAG = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.ChiralType.values.values())
}

DICT_HYBRIDIZATION = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.HybridizationType.values.values())
}

DICT_BOND_DIR = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondDir.values.values())
}

DICT_BOND_TYPE = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondType.values.values())
}

DICT_BOND_STEREO = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondStereo.values.values())
}

ATOM_CATE_FEAT_DIMS = [53, 3, 8, 7, 5, 5, 7, 2, 2, 7, 5, 90, 10, 2, 5, 5]
ATOM_FLOAT_FEAT_DIM = 1
BOND_CATE_FEAT_DIMS = [5, 13, 2, 2, 2, 4]
BOND_FLOAT_FEAT_DIM = 2

MAX_SAME_RING_COUNT = 26
MAX_SAME_RING_MIN_SIZE = 22
MAX_SHORTEST_PATH_LEN = 41
MAX_ANGLES = 180 + 3


def extract_mol_xyz(mol: rdkit.Chem.Mol):
    '''use rdkit to extract 3d coordinates

    Returns:
        xyz: (num_atom, 3),
        mol: rdkit.Chem.Mol, because embedding process may change the order of the mol
    '''

    mol2 = rdkit.Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol2, useExpTorsionAnglePrefs=True,useBasicKnowledge=True) != 0:
            failEmbed = True
            pass
        else:
            try:
                AllChem.UFFOptimizeMolecule(mol2)
            except rdkit.Chem.rdchem.KekulizeException:
                print('exception happened in UFFOptimizeMolecule')
                mol2 = rdkit.Chem.AddHs(mol)
                rdkit.Chem.AllChem.EmbedMolecule(mol2)
                pass
            failEmbed = False
            pass
        pass
    except Exception as e:
        print(e)
        failEmbed = True
        pass

    if failEmbed:
        num_atom = mol.GetNumAtoms()
        return -np.ones((num_atom, 3), dtype='float32'), mol
    else:
        mol2 = rdkit.Chem.RemoveHs(mol2)
        xyz = mol2.GetConformer().GetPositions()
        return np.asarray(xyz, dtype='float32'), mol2
    pass


def extract_mol_xy(mol: rdkit.Chem.Mol):
    AllChem.Compute2DCoords(mol)
    xyz = np.asarray(
        mol.GetConformer().GetPositions(),
        dtype='float32')

    return xyz[:, :2]


def extract_atom_cate_feat(mol: rdkit.Chem.Mol):

    feat = np.zeros((mol.GetNumAtoms(), DIM_ATOM_CATE_FEAT), dtype='int32')
    
    for i in range(mol.GetNumAtoms()):
        atom: rdkit.Chem.Atom
        atom = mol.GetAtomWithIdx(i)

        feat[i, 0] = atom.GetAtomicNum() - 1
        feat[i, 1] = DICT_CHIRALTAG[atom.GetChiralTag()]
        feat[i, 2] = atom.GetTotalDegree()
        feat[i, 3] = atom.GetDegree()
        feat[i, 4] = atom.GetTotalNumHs()
        feat[i, 5] = atom.GetNumRadicalElectrons()
        feat[i, 6] = DICT_HYBRIDIZATION[atom.GetHybridization()]
        feat[i, 7] = int(atom.GetIsAromatic())
        feat[i, 8] = int(atom.IsInRing())
        feat[i, 9] = atom.GetExplicitValence()
        feat[i, 10] = atom.GetImplicitValence()
        feat[i, 11] = atom.GetIsotope()
        feat[i, 12] = atom.GetFormalCharge() + 5
        feat[i, 13] = int(atom.GetNoImplicit())
        feat[i, 14] = atom.GetNumExplicitHs()
        feat[i, 15] = atom.GetNumImplicitHs()
        
        pass

    return feat
    pass


def extract_atom_float_feat(mol: rdkit.Chem.Mol):
    
    feat = np.zeros((mol.GetNumAtoms(), DIM_ATOM_FLOAT_FEAT), dtype='float32')
    for i, atom in enumerate(mol.GetAtoms()):
        feat[i, 0] = 1.0 / atom.GetMass()
        pass

    return feat
    pass


def extract_bond_feat(mol: rdkit.Chem.Mol):
    num_bonds = mol.GetNumBonds()
    if num_bonds == 0:
        bond_index = np.empty((2, 0), dtype='int32')
        bond_cate_feat = np.empty((0, DIM_BOND_CATE_FEAT), dtype='int32')
        bond_float_feat = np.empty((0, DIM_BOND_FLOAT_FEAT), dtype='float32')
        pass
    else:
        bond_index = np.zeros((2, num_bonds), dtype='int32')
        bond_cate_feat = np.zeros((num_bonds, DIM_BOND_CATE_FEAT),
                                  dtype='int32')
        bond_float_feat = np.zeros((num_bonds, DIM_BOND_FLOAT_FEAT),
                                   dtype='float32')

        for ib, bond in enumerate(mol.GetBonds()):
            bond: rdkit.Chem.Bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
        
            bond_index[0, ib] = i            
            bond_index[1, ib] = j

            bond_cate_feat[ib, 0] = DICT_BOND_DIR[bond.GetBondDir()]
            bond_cate_feat[ib, 1] = DICT_BOND_TYPE[bond.GetBondType()]
            bond_cate_feat[ib, 2] = int(bond.GetIsAromatic())
            bond_cate_feat[ib, 3] = int(bond.GetIsConjugated())
            bond_cate_feat[ib, 4] = int(bond.IsInRing())
            bond_cate_feat[ib, 5] = DICT_BOND_STEREO[bond.GetStereo()]

            bond_float_feat[ib, 0] = bond.GetValenceContrib(
                bond.GetBeginAtom())
            bond_float_feat[ib, 1] = bond.GetValenceContrib(
                bond.GetEndAtom())
            pass
        pass
    
    return bond_index, bond_cate_feat, bond_float_feat


def extract_mol_feat_cate(mol: rdkit.Chem.Mol):
    fea = np.zeros(4, dtype='float32')
    fea[0] = mol.GetNumAtoms()
    fea[1] = mol.GetNumBonds()
    fea[2] = mol.GetNumHeavyAtoms()
    ringinfo = mol.GetRingInfo()
    fea[3] = ringinfo.NumRings()
    return fea


def extrac_mol_fingerprint(mol: rdkit.Chem.Mol):

    fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3))

    return np.argwhere(fp).flatten().astype('int32')


def atom_same_ring_info(mol):
    '''calculate same ring infomation

    Returns:
        same_ring_count: (num_atom, num_atom), the number of rings shared by each pair of atoms
        same_ring_min_size: (num_atom, num_atom), the minimum size of rings shared by each pair of atoms
    '''
    ringInfo = mol.GetRingInfo()

    atom_rings = ringInfo.AtomRings()

    num_atoms = mol.GetNumAtoms()
    num_rings = len(atom_rings)

    if num_rings == 0:
        return np.zeros((num_atoms, num_atoms), dtype='float32'), \
            np.zeros((num_atoms, num_atoms), dtype='float32')

    atom_groups = np.zeros((num_atoms, num_rings), dtype='float32')
    for i, ring_atoms in enumerate(atom_rings):
        atom_groups[ring_atoms, i] = 1
        pass

    ring_sizes = np.array(
        [len(r) for r in atom_rings], dtype='float32').reshape(1, 1, -1)

    is_same_ring = atom_groups[:, None, :] * atom_groups[None, :, :]

    same_ring_count = np.sum(is_same_ring, axis=-1)
    # is_same_ring[same_ring_count==0, :] = 999

    is_same_ring[(is_same_ring == 0) & ((same_ring_count>0)[..., None])] = 999
    same_ring_min_size = np.min(is_same_ring * ring_sizes, axis=-1)

    
    return same_ring_count, same_ring_min_size


def path_node2edge(nodes, edge_dict):

    edge_path = []
    for i in range(1, len(nodes)):
        eid = edge_dict.get((nodes[i-1], nodes[i]))
        if eid is None:
            eid = edge_dict[(nodes[i], nodes[i-1])]
            pass
        edge_path.append(eid)

        pass

    return edge_path
    pass

def shortest_path_length(edge_index, num_nodes, xyz):
    '''calculate shortest path length between each pair of atoms

    Returns:
        dist: path length matrix (num_nodes, num_nodes)
        paths: path of edge index
        pathsAtom: path of atom index
    '''
    dist = -np.ones((num_nodes, num_nodes), dtype='int16')
    angles = np.zeros((num_nodes, num_nodes), dtype='int16')

    paths = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]
    pathsAtom = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]

    if edge_index.shape[1] > 0:
        g = networkx.Graph()
        g.add_edges_from(
            edge_index.T
        )
        
        edge_dict = {
            (edge_index[0, i], edge_index[1, i]): i for i in range(
                edge_index.shape[1])
        }
        
        res = networkx.all_pairs_shortest_path(g)

        for sid, path_dict in res:
            for did, path in path_dict.items():
                if len(path) == 1:
                    dist[sid, did] = 0
                else:
                    dist[sid, did] = len(path) - 1
                    # edge_path = path_node2edge(path, edge_dict)
                    # paths[sid][did].update(edge_path)
                    # pathsAtom[sid][did].update(path[1:-1])

                    if len(path) == 3:
                        mid = path[1]
                        vd = xyz[did] - xyz[mid]
                        vs = xyz[sid] - xyz[mid]
                        vd_norm = np.linalg.norm(vd)
                        vs_norm = np.linalg.norm(vs)
                        if vd_norm == 0 or vs_norm == 0:
                            angles[sid, did] = 1
                        else:
                            cosine = np.dot(vd, vs) / (vd_norm * vs_norm)
                            cosine = max(-1, min(1, cosine))
                            degree = np.arccos(cosine) * 180 / np.pi
                            angles[sid, did] = int(degree) + 2
                            pass
                        pass
                    
                    if len(path) > 2:
                        paths[sid][did] = path[1:-1]
                        pass
                    pass
                pass
            pass
        pass
    
    else:
        pass
    
    return dist.astype('int16') + 1, angles, paths

def smiles2graph(s):

    mol = rdkit.Chem.MolFromSmiles(s)
    # xyz, mol = extract_mol_xyz(mol)

    return mol2graph(mol)

def mol2graph(mol: rdkit.Chem.rdchem.Mol):
    if mol.GetNumConformers() > 0:
        xyz = mol.GetConformer().GetPositions().astype('float32')
    else:
        xyz = np.zeros((mol.GetNumAtoms(), 3), dtype='float32')
        pass

    atom_float_feat = extract_atom_float_feat(mol)
    atom_cate_feat = extract_atom_cate_feat(mol)

    bond_index, bond_cate_feat, bond_float_feat = \
        extract_bond_feat(mol)

    graph = dict()
    atom_same_ring_count, atom_same_ring_min_size = atom_same_ring_info(mol)
    graph['atom_same_ring_count'] = atom_same_ring_count
    graph['atom_same_ring_min_size'] = atom_same_ring_min_size

    # atom_cate_feat = np.hstack(
    #     (atom_cate_feat, graph['atom_same_ring_count'].diagonal().reshape(
    #         -1, 1).astype('int32') - 1))
    
    graph['bond_index'] = bond_index
    graph['bond_feat_cate'] = bond_cate_feat
    graph['bond_feat_float'] = bond_float_feat
    graph['atom_feat_cate'] = atom_cate_feat
    graph['atom_feat_float'] = atom_float_feat
    graph['xyz'] = xyz
    graph['num_atoms'] = atom_cate_feat.shape[0]
    graph['num_bonds'] = bond_index.shape[1]
    dist, angles, paths = shortest_path_length(
        bond_index, atom_cate_feat.shape[0], xyz)

    graph['shortest_path_length'] = dist
    graph['angles'] = angles
    graph['shortest_path'] = paths
    
    # graph['shortest_path'] = paths
    # graph['shortest_path_atom'] = pathsAtom
    
    # graph['graph_feat_cate'] = features.extract_mol_feat_cate(mol)
    # graph['graph_fp'] = features.extrac_mol_fingerprint(mol)
    # graph['graph_fp_size'] = 2048

    return graph    
    pass
