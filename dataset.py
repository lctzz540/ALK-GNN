from torch_geometric.data import InMemoryDataset, Data, Batch
import torch
import os
import pandas as pd
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
import numpy as np


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        filename="Data.csv",
        transform=None,
        pre_transform=None,
    ):
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return [
            "data.pt",
        ]

    def download(self):
        pass

    def process(self):
        self.datadf = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        data = []

        for index, row in tqdm(self.datadf.iterrows(), total=len(self.datadf)):
            mol = Chem.MolFromSmiles(row["Canomicalsmiles"])
            f = featurizer._featurize(mol)
            node_features = torch.tensor(f.node_features, dtype=torch.float)
            edge_index = torch.tensor(f.edge_index, dtype=torch.int64)
            edge_features = torch.tensor(f.edge_features, dtype=torch.float)

            label = self._get_labels(row["pChEMBL"])
            data_point = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                y=label,
                smiles=row["Canomicalsmiles"],
            )
            data.append(data_point)

        data = Batch.from_data_list(data)

        torch.save(data, os.path.join(self.processed_dir, "data.pt"))

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(
            label, dtype=torch.float32
        )  # Adjust the data type as needed

    def len(self):
        data_file = os.path.join(self.processed_dir, "data.pt")
        if os.path.exists(data_file):
            data = torch.load(data_file)
            return len(data)
        return 0

    def get(self, idx):
        data_file = os.path.join(self.processed_dir, "data.pt")
        if os.path.exists(data_file):
            data = torch.load(data_file)
            return data[idx]
        else:
            self.process()
            return self.get(idx)
