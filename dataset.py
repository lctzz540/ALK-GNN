import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.utils import one_hot, scatter
from rdkit.Chem import AllChem
import torch_sparse


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        self.root = root
        self.filename = filename
        self.data = self.get()
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def _generateMolFromSmiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=0xF00D)
        AllChem.MMFFOptimizeMolecule(m)
        return m

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        data_list = []

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol = self._generateMolFromSmiles(row["Canomicalsmiles"])
            except:
                continue
            label = self._get_labels(row["pChEMBL"])
            smiles = row["Canomicalsmiles"]
            types = {
                "H": 0,
                "C": 1,
                "N": 2,
                "O": 3,
                "F": 4,
                "Cl": 5,
                "S": 6,
                "Br": 7,
                "I": 8,
            }
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

            N = mol.GetNumAtoms()
            pos = mol.GetConformer().GetPositions()
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            if not isinstance(edge_index, torch.LongTensor):
                edge_index = edge_index.to(torch.long)
                edge_index = edge_index.t()

            if not isinstance(edge_index, torch_sparse.SparseTensor):
                edge_index = torch_sparse.SparseTensor(
                    edge_index, torch.ones(edge_index.shape[1])
                )

            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            hs = (z == 1).to(torch.float)
            num_hs = scatter(
                hs[edge_index[0]], edge_index[1], dim_size=N, reduce="sum"
            ).tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2,
                        sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            data = Data(
                x=x,
                z=z,
                pos=torch.tensor(pos, dtype=torch.float),
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=label,
                name=None,
                idx=index,
            )

            data_list.append(data)
        torch.save(data_list, os.path.join(self.processed_dir, "data.pt"))

    def _get_labels(self, label):
        label = [label]
        return torch.tensor(label, dtype=torch.float)

    def len(self):
        return len(self.data)

    def get(self, idx=None):
        data_file = os.path.join(self.processed_dir, "data.pt")
        if not os.path.exists(data_file):
            self.process()
        return torch.load(data_file)
