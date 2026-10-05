"""RedditWalk dataset: random walk with restart subgraphs of the Reddit GWCC (report sec 6.3).

Source: origin/case-study-v2:masked-diff-graph/case_study/reddit_dataset.py (7537553).
Changes: save to `root` (not a hardcoded path), save Jaccard stats and node lists next to the
`.pt` file, and raise an error when the walk cannot reach the target size. Sampling is unchanged.

Keep this module free of Python 3.10+ syntax, so the Python 3.9 sparserdiff env can import it.
"""

import json
import os
import random
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RedditDataset(Dataset):

    def __init__(self, stage='train', root='data/redditwalk/subgraphs', dataset_file=None,
                 creation_mode=False, csv_path=None,
                 num_subgraphs=80, subgraph_size=800, base_seed=42, p_restart=0.3):

        self.stage = stage
        self.root = root

        os.makedirs(root, exist_ok=True)

        # Node type = BERT cluster id + 2. Labels are the only record of the 20 topics.
        self.types = {
            "MASK": 0,
            "PAD": 1,
            "General Discussion & Lifestyle": 2,
            "TV, Film & Entertainment Fans": 3,
            "Local & Geographic Communities": 4,
            "Anime & Manga": 5,
            "Meta & Default Communities": 6,
            "Hobby & Special Interest": 7,
            "Sports Teams & Fan Groups": 8,
            "Dedicated Niche Communities": 9,
            "Community Q&A & Sharing": 10,
            "News & Political Discussion": 11,
            "Music & Audio": 12,
            "Crypto & Technology": 13,
            "International (Non-English)": 14,
            "Gaming & Esports": 15,
            "Links & External Resources": 16,
            "Social Issues & Current Events": 17,
            "Casual, Meme & Humor": 18,
            "Creative Writing & Fiction": 19,
            "Visual Art & Media": 20,
            "Minecraft & Game Servers": 21
        }

        self.charges = {-999: 0, -998: 1, 0: 2}

        self.bonds = {"MASK": 0, "NO_BOND": 1, "Positive": 2, "Negative": 3}

        self._records = []

        if creation_mode:
            if csv_path is None:
                raise ValueError("path to giant_component.csv must be provided when creation_mode=True")
            preprocessed_path = self.preprocess(csv_path)
            GCC = self.build_graph(preprocessed_path)

            print('Sampling Graphs...')
            node_lists = []
            for i in tqdm(range(num_subgraphs)):
                subgraph = self.sample_subgraph(GCC, p_restart=p_restart,
                                                max_size=subgraph_size, seed=i + base_seed)
                node_lists.append(list(subgraph.nodes()))
                A, C, E = self.nx_to_ACE(subgraph)
                self._records.append((A, C, E))

            stem = f'{num_subgraphs}_reddit_subgraphs_{subgraph_size}'
            dataset_name = stem + '.pt'
            torch.save(self._records, os.path.join(self.root, dataset_name))
            print(f'Dataset Created: {dataset_name}')

            # compute pairwise Jaccard (Table 16)
            node_sets = [set(nodes) for nodes in node_lists]
            jaccard_scores = []
            for s1, s2 in combinations(node_sets, 2):
                j = len(s1 & s2) / len(s1 | s2)
                jaccard_scores.append(j)

            print(f"Mean Jaccard: {np.mean(jaccard_scores):.4f}")
            print(f"Max Jaccard: {np.max(jaccard_scores):.4f}")

            # Save the numbers and node lists, so Table 16 can be recomputed from files.
            jaccard = {
                'num_subgraphs': num_subgraphs,
                'subgraph_size': subgraph_size,
                'base_seed': base_seed,
                'p_restart': p_restart,
                'mean': float(np.mean(jaccard_scores)),
                'max': float(np.max(jaccard_scores)),
                'n_pairs': len(jaccard_scores),
            }
            with open(os.path.join(self.root, stem + '_jaccard.json'), 'w') as f:
                json.dump(jaccard, f, indent=2)
            # Node names in sample order. Order matches the rows of A and E in the .pt file.
            with open(os.path.join(self.root, stem + '_nodes.json'), 'w') as f:
                json.dump(node_lists, f)

        else:
            if dataset_file is None:
                raise ValueError("dataset_file must be provided when creation_mode=False")
            self._records = torch.load(os.path.join(self.root, dataset_file), weights_only=False)

        # train/test split
        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(len(self._records), generator=g)

        test_len = int(round(len(self._records) * 0.2))
        train_len = len(self._records) - test_len

        if stage == 'train':
            self._records = [self._records[i] for i in indices[:train_len]]
        else:
            self._records = [self._records[i] for i in indices[train_len:]]

        self.max_length = len(self._records[0][0])

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def preprocess(self, csv_path):
        """Keep only edges whose both ends have a description (cluster id != -1)."""
        df = pd.read_csv(csv_path)

        df = df[(df['source_type'] != -1) & (df['target_type'] != -1)]

        # drop some columns that won't be used in final dataset
        df = df.drop(columns=['POST_ID', 'TIMESTAMP', 'PROPERTIES', 'source_topic', 'target_topic'])

        output_path = os.path.join(os.path.dirname(csv_path), 'giant_preprocess.csv')

        df.to_csv(output_path, index=False)

        print(len(df))

        return output_path

    def build_graph(self, csv_path):
        print('Building GWCC...')

        df = pd.read_csv(csv_path)

        G = nx.from_pandas_edgelist(df,
                                    'SOURCE_SUBREDDIT',
                                    'TARGET_SUBREDDIT',
                                    edge_attr=['LINK_SENTIMENT'],
                                    create_using=nx.Graph())

        for _, row in df.iterrows():
            G.nodes[row['SOURCE_SUBREDDIT']]['topic'] = row['source_type']
            G.nodes[row['TARGET_SUBREDDIT']]['topic'] = row['target_type']

        return G

    def sample_subgraph(self, Graph, p_restart=0.3, max_size=800, seed=None):
        """
        Random Walk with Restart Subgraph Sampling

        Samples a topic, chooses one seed of that topic

        Random Walk with Restart until max_size
        """
        if seed is not None:
            random.seed(seed)

        topics = list(set(nx.get_node_attributes(Graph, 'topic').values()))
        chosen_topic = random.choice(topics)

        candidates = [n for n, d in Graph.nodes(data=True) if d.get('topic') == chosen_topic]
        start = random.choice(candidates)

        # Termination check: the walk can only reach nodes in the start node's component.
        # This uses no random numbers, so successful runs are unchanged.
        component_size = len(nx.node_connected_component(Graph, start))
        if component_size < max_size:
            raise ValueError(
                f"seed={seed}: start node {start!r} is in a component of {component_size} nodes, "
                f"fewer than max_size={max_size}. The walk would never end.")

        current = start
        visited = {start}

        while len(visited) < max_size:
            if random.random() < p_restart:
                current = start
            else:
                neighbors = list(Graph.neighbors(current))
                if neighbors:
                    current = random.choice(neighbors)
            visited.add(current)

        subgraph = Graph.subgraph(visited).copy()

        return subgraph

    def nx_to_ACE(self, subgraph):
        nodes = list(subgraph.nodes())
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}

        A = [subgraph.nodes[n]['topic'] + 2 for n in nodes]

        C = [self.charges[0]] * N

        E = np.full((N, N), self.bonds["NO_BOND"], dtype=np.int64)

        for u, v, data in subgraph.edges(data=True):
            sentiment = data["LINK_SENTIMENT"]
            bond_type = self.bonds["Positive"] if sentiment > 0 else self.bonds['Negative']

            E[node_idx[u]][node_idx[v]] = bond_type
            E[node_idx[v]][node_idx[u]] = bond_type

        return A, C, E

    def compute_M_max(self):
        M_max = 0
        for A, C, E in self._records:
            edges = int((E > 1).sum() // 2)
            M_max = max(M_max, edges)
        print(f"M_max: {M_max}")
        return M_max
