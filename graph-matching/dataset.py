import copy
import random
from typing import Tuple

import networkx as nx
import numpy as np

from torch_geometric.data import Dataset, DataLoader
from torch_geometric.utils import from_networkx


class SyntheticGraphDataset(Dataset):
    def __init__(self):
        self.min_nodes = 19
        self.max_nodes = 20
        self.min_edge_p = 0.2
        self.max_edge_p = 0.2
        self.dataset_size = 10

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def __len__(self):
        return self.dataset_size

    def _generate_graph(self) -> nx.Graph:
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)
        p_edge = np.random.uniform(self.min_edge_p, self.max_edge_p)

        # do a little bit of filtering
        n_trials = 100
        for _ in range(n_trials):
            g = nx.erdos_renyi_graph(n_nodes, p_edge)
            if nx.is_connected(g):
                return g

        raise ValueError("Failed to generate a connected graph.")

    def get(self, idx: int) -> Tuple:
        seed = 13 + idx
        np.random.seed(seed)
        random.seed(seed)
        g, g_other = self.get_pair(idx // 2 == 0)
        return from_networkx(g), from_networkx(g_other)

    @staticmethod
    def perturb_graph(g: nx.Graph, n: int) -> nx.Graph:
        """Substitutes n edges from graph g with another n randomly picked edges."""
        _g = copy.deepcopy(g)
        n_nodes = _g.number_of_nodes()
        edges = list(_g.edges())
        # sample n edges without replacement
        e_remove = [
            edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)
        ]
        edge_set = set(edges)
        e_add = set()
        while len(e_add) < n:
            e = np.random.choice(n_nodes, 2, replace=False)
            # make sure e does not exist and is not already chosen to be added
            if (
                (e[0], e[1]) not in edge_set
                and (e[1], e[0]) not in edge_set
                and (e[0], e[1]) not in e_add
                and (e[1], e[0]) not in e_add
            ):
                e_add.add((e[0], e[1]))

        for i, j in e_remove:
            _g.remove_edge(i, j)
        for i, j in e_add:
            _g.add_edge(i, j)
        return _g

    def get_pair(self, positive: bool) -> Tuple:
        g = self._generate_graph()
        n_changes = 1 if positive else 2
        perturbed_g = self.perturb_graph(g, n_changes)
        return g, perturbed_g

    def __getitem__(self, index):
        return self.get(index)


class SyntheticDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle)
