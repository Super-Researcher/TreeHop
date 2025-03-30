import os
import re
from typing import Iterable, Union
import dgl
import networkx as nx
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset

from utils import NodeType
from src.normalize_text import normalize
from src.evaluation import normalize_unicode, white_space_fix


class TreeHopTrainDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        *,
        negative_dataset: str = None,
        exclude_comparison=True,
        num_negatives: int = 4,
        graph_cache_dir: str = None,
        device=None
    ):
        super().__init__()
        self._re_clean_title = re.compile(r"\(.*\)")

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.negative_dataset = negative_dataset
        self.exclude_comparison = exclude_comparison
        self.num_negatives = num_negatives
        self.graph_cache_dir = graph_cache_dir
        self.device = device

        self.ary_contexts = np.load(f"embedding_data/{dataset_name}/{dataset_type}_dense.npy")
        if negative_dataset is not None:
            ary_neg_contexts = np.load(negative_dataset)
            # negative indices will start from len(self.ary_contexts)
            self.idx_neg_start = len(self.ary_contexts)
            self.ary_contexts = np.concatenate([self.ary_contexts, ary_neg_contexts], axis=0)

        self.ary_contexts = torch.as_tensor(self.ary_contexts, dtype=torch.float32, device=device)
        self.df_dataset = self.load_dataset(dataset_name, dataset_type)

        if self.graph_cache_dir is None:
            self._create_graphs(
                self.df_dataset, self.ary_contexts,
                num_negatives=self.num_negatives,
                device=self.device
            )
        else:
            self.dataset_path = os.path.join(
                self.graph_cache_dir, f"{self.dataset_name}_{self.dataset_type}_df_dataset.pkl"
            )
            self.graph_cache_path = os.path.join(
                self.graph_cache_dir, f"{self.dataset_name}_{self.dataset_type}_dgl_graph.bin"
            )
            if self.has_graph_cache():
                self.load_graph_cache()
            else:
                self._create_graphs(
                    self.df_dataset, self.ary_contexts,
                    num_negatives=self.num_negatives,
                    device=self.device
                )
                self.save_graph_cache()

    def load_dataset(self, dataset_name: str, dataset_type: str):
        if dataset_type == "train":
            df_dataset = pd.read_json(
                f"train_data/{dataset_name}_train_processed.jsonl", lines=True, orient="records"
            )
            # ary_contexts = np.load(f"embedding_data/{dataset_name}/train_dense.npy")
        elif dataset_type == "eval":
            df_dataset = pd.read_json(
                f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True, orient="records"
            )
        else:
            raise LookupError(f"cannot find {dataset_type} {dataset_name}")

        return df_dataset

    def clean_title(self, title: str):
        # title = re.sub(self._re_clean_title, '', title)
        return normalize_unicode(white_space_fix(normalize(title).strip()))

    def _match_evidence_title(self, evidences, supporting_facts, ctxs):
        is_match = True
        for evi in evidences:
            if evi[0] not in supporting_facts:
                is_match = False
                for fact in supporting_facts:
                    if self.clean_title(evi[0]) == self.clean_title(fact):
                        evi[0] = fact
                        is_match = True

                if not is_match:
                    for fact in supporting_facts:
                        if self.clean_title(evi[0]) in self.clean_title(fact):
                            evi[0] = fact
                            is_match = True

                if not is_match:
                    for ctx in ctxs:
                        if evi[0] == ctx["title"] or ctx["text"].startswith(evi[0]):
                            evi[0] = ctx["title"]
                            is_match = True

            # if not is_match:
            #     raise LookupError(f"Evidence name '{evi[0]}' not found in {supporting_facts}")

            if evi[2] not in supporting_facts:
                is_match = False
                cleaned_evi = self.clean_title(evi[2])
                for fact in supporting_facts:
                    cleaned_fact = self.clean_title(fact)
                    if cleaned_evi == cleaned_fact:
                        evi[2] = fact
                        is_match = True

                if not is_match:
                    for fact in supporting_facts:
                        cleaned_fact = self.clean_title(fact)
                        if cleaned_evi.endswith(cleaned_fact) or cleaned_evi.startswith(cleaned_fact) \
                            or cleaned_fact.endswith(cleaned_evi) or cleaned_fact.startswith(cleaned_evi):
                            evi[2] = fact
                            is_match = True

                if not is_match:
                    for ctx in ctxs:
                        if evi[2] == ctx["title"] or ctx["text"].startswith(evi[2]):
                            evi[2] = ctx["title"]
                            is_match = True
            # if not is_match:
            #     raise LookupError(f"Evidence name '{evi[1]}' not found in {supporting_facts}")

    def _gen_negative_samples(
            self,
            num_samples: int,
            num_positives: int,
            num_negatives: int,
            exclude: Union[set, Iterable]
        ):
        """generate positive and negative samples for specific title set

        Args:
            titles (Iterable[str]): titles to 
            num_samples (int): number of selectable samples in total
            num_negatives (int): number of negatives to generate for each positive sample
            excluded (set): index of context to exclude

        Returns:
            indices of positive samples and their corresponding list of indices of negative samples
        """
        if not isinstance(exclude, set):
            exculde = set(exclude)

        if self.negative_dataset is None:
            lst_choices = [n for n in range(num_samples) if n not in exculde]
        else:
            lst_choices = list(range(self.idx_neg_start, len(self.ary_contexts)))

        lst_negatives: list[list[int]] = []
        for _ in range(num_positives):
            ary_negatives = np.random.choice(
                lst_choices,
                size=num_negatives,
                replace=False
            )

            lst_negatives.append(ary_negatives.tolist())

        return lst_negatives

    def graph_propagator(self, index: int, num_negatives=5):
        """generate indices of positive and negative samples in breath-first search order

        Args:
            index (int): index of record in dataset
            num_negatives (int, optional): number of negative samples. Defaults to 5.

        Yields:
            list of list of indices of positive samples and corresponding list of negative samples.
        """
        ctxs = self.df_dataset.loc[index, "ctxs"]
        evidences = self.df_dataset.loc[index, "evidences"]
        supporting_facts = {fact[0] for fact in self.df_dataset.loc[index, "supporting_facts"]}
        self._match_evidence_title(evidences, supporting_facts, ctxs)

        # strict rule
        set_titles = set(src for src, _, dst in evidences)
        d_title2idx = {title: n
                        for title in set_titles for n, ctx in enumerate(ctxs)
                        if ctx["title"] in title}
        if len(d_title2idx) < len(set_titles):
            # relax rule
            d_title2idx = {title: n
                           for title in set_titles for n, ctx in enumerate(ctxs)
                           if ctx["title"] in title or ctx["text"].startswith(title)}

        assert len(d_title2idx) == len(set_titles), f"{set_titles}\n{[ctx['title'] for ctx in ctxs]}"

        lst_empty = list()
        d_nodes = dict()
        set_last_layer_nodes = set()
        for evi in evidences:
            idx_src, idx_dst = d_title2idx[evi[0]], d_title2idx.get(evi[2], None)
            if idx_dst is None:
                # reaches last layer
                set_last_layer_nodes.add(idx_src)
                if idx_src not in d_nodes:
                    d_nodes[idx_src] = lst_empty
                continue

            elif idx_src in d_nodes and idx_dst not in d_nodes[idx_src]:
                d_nodes[idx_src].append(idx_dst)
            else:
                d_nodes[idx_src] = [idx_dst]

        assert len(d_nodes) > 1, "Detect single-hop graph"
        g = nx.DiGraph(d_nodes)
        assert self.is_acyclic(g), \
            "Detect loop in the graph: the same context has been used more than once."

        set_start_nodes = set(d_nodes.keys())
        set_end_nodes = set(itertools.chain(*d_nodes.values()))

        # first_layer nodes
        set_first_layer_nodes = set_start_nodes - set_end_nodes

        query_type = self.df_dataset.loc[index, "type"]
        if query_type == 'compositional' or query_type == 'inference':
            assert len(set_first_layer_nodes) == 1 and len(set_last_layer_nodes) > 0, \
                f"Not a {query_type}"
        elif query_type == 'comparison':
            assert len(set_first_layer_nodes) > 1 and (set_first_layer_nodes == set_last_layer_nodes), \
                f"Not a {query_type}"
        elif query_type == 'bridge_comparison':
            assert len(set_first_layer_nodes) > 1 and len(set_last_layer_nodes) > 1, \
                f"Not a {query_type}"

        # BFS
        current: list[list[int]] = [list(set_first_layer_nodes)]
        while any(len(nodes) > 0 for nodes in current):
            # exclude current and its children as they are positives
            #TODO: should we exclude any visited (negative) nodes?
            lst_current = list(itertools.chain(*current))
            lst_current_negatives: list[list[list[int]]] = []
            for positives in current:
                lst_children = list(itertools.chain(*(
                    d_nodes.get(node, lst_empty)
                    for node in positives)
                ))

                negatives: list[list[int]] = self._gen_negative_samples(
                    num_positives=len(positives),
                    num_samples=len(ctxs),
                    num_negatives=num_negatives,
                    exclude=lst_current + lst_children
                )
                lst_current_negatives.append(negatives)

            yield current, lst_current_negatives

            current = [d_nodes.get(node, lst_empty) for node in lst_current]

    def _make_comparison_trainable(self, index, lst_graph_idx):
        query_type = self.df_dataset.loc[index, "type"]
        if "comparison" not in query_type or len(lst_graph_idx) >= 2:
            return lst_graph_idx

        (lst_query_idx,), (lst_ctx_idx,) = lst_graph_idx[0]
        assert len(lst_query_idx) == 2, "only works when comparing two entities."
        lst_graph_idx.append((
            [[idx] for idx in reversed(lst_query_idx)], [[idx] for idx in lst_ctx_idx]
        ))
        return lst_graph_idx

    def _create_graph_helper(self, index, row, ary_contexts, num_negatives, device):
        if self.exclude_comparison and row["type"] == "comparison":  # row["type"] != "compositional"
            return index

        try:
            lst_idx_graph = list(self.graph_propagator(index, num_negatives))
            lst_idx_graph = self._make_comparison_trainable(index, lst_idx_graph)
            # print(i, idx_positive_nodes, idx_negative_nodes)
        except AssertionError:
            return index
        except TypeError as e:
            print(index, row["evidences"][0], row["evidences"][0][0])
            return index

        lst_last_layer_pos = []
        idx_current = 1
        lst_y: list[NodeType] = [NodeType.query.value]
        prev_pos = [0]
        lst_node_out = []
        lst_node_in = []
        lst_idx_ctxs = [index]  # query node
        # propagate in BFS order
        for lst_idx_pos, lst_idx_negs in lst_idx_graph:
            assert len(prev_pos) == len(lst_idx_pos) == len(lst_idx_negs), \
                "number of nodes in new layer mismatches with the last layer"
            # layer-wise
            lst_tmp = []
            for pp, idx_pos, idx_negs in zip(prev_pos, lst_idx_pos, lst_idx_negs):
                if len(idx_pos) == 0:
                    lst_last_layer_pos.append(pp)
                    continue

                lst_y.extend([
                    NodeType.relevant_doc.value,
                    *([NodeType.irrelevant_doc.value] * num_negatives)
                ] * len(idx_pos))
                # assign node index
                n_current_nodes = len(idx_pos) * (1+num_negatives)
                lst_current = list(range(idx_current, idx_current + n_current_nodes))

                for n, (i_pos, lst_i_negs) in enumerate(zip(idx_pos, idx_negs)):
                    lst_node_out.extend([pp] * (1+num_negatives))
                    lst_node_in.extend(lst_current[n * (1+num_negatives): (n+1) * (1+num_negatives)])
                    if self.negative_dataset is None:
                        lst_idx_ctxs.extend([row["ctxs"][idx]["idx"] for idx in [i_pos] + lst_i_negs])
                    else:
                        lst_idx_ctxs.extend([row["ctxs"][i_pos]["idx"]] + lst_i_negs)

                lst_tmp.extend(lst_current[:: 1+num_negatives])
                idx_current += n_current_nodes

            prev_pos = lst_tmp

        # label last layer positive nodes to pesudo query node
        lst_last_layer_pos.extend(prev_pos)
        y = torch.as_tensor(lst_y, device=device)
        y[lst_last_layer_pos] = NodeType.leaf.value

        graph = dgl.graph(
            (lst_node_out, lst_node_in),
            idtype=torch.int32,
            num_nodes=idx_current,
            device=device
        )

        graph.ndata["y"] = y
        #TODO: Check rep alignment
        graph.ndata["rep"] = ary_contexts[lst_idx_ctxs]
        # h_0 = rep_q
        graph.ndata["h"] = ary_contexts[index].repeat(graph.num_nodes(), 1)
        return graph

    def _create_graphs(
        self,
        df_dataset: pd.DataFrame,
        ary_contexts: np.ndarray,
        num_negatives: int = 5,
        device=None,
        num_workers=8
    ):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            tasks = {
                executor.submit(
                    self._create_graph_helper,
                    index, row, ary_contexts, num_negatives, device
                ): index
                for index, row in df_dataset.iterrows()
            }
            lst_skip_index = []
            results = []
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Creating graphs"):
                idx = tasks[future]
                res = future.result()
                if isinstance(res, int):
                    lst_skip_index.append(res)
                    continue

                results.append((idx, res))

        self._graphs = [res[1] for res in sorted(results, key=lambda x: x[0])]
        self.df_dataset.drop(index=lst_skip_index, inplace=True)
        print(f"Skipped {lst_skip_index} records. Total trainable: {len(self.df_dataset)}")
        return self._graphs

    @property
    def graphs(self):
        if hasattr(self, "_graphs"):
            return self._graphs

        return self._create_graphs(
            self.df_dataset, self.ary_contexts,
            num_negatives=self.num_negatives,
            device=self.device
        )

    def save_graph_cache(self):
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        # save graphs and labels
        dgl.save_graphs(self.graph_cache_path, self.graphs)

        # save dataset and other information
        self.df_dataset.to_pickle(self.dataset_path)

    def load_graph_cache(self):
        # load processed data from directory `graph_cache_path`
        print(f"Loading graph cache '{self.graph_cache_path}'")
        self._graphs, _ = dgl.load_graphs(self.graph_cache_path)

        # load dataset and other information
        print(f"Loading dataset '{self.dataset_path}'")
        self.df_dataset = pd.read_pickle(self.dataset_path)
        self.to(self.device)

    def has_graph_cache(self):
        return os.path.exists(self.graph_cache_path) \
            and os.path.exists(self.dataset_path)

    @classmethod
    def is_acyclic(cls, graph: Union[dgl.DGLGraph, nx.Graph]):
        if isinstance(graph, dgl.DGLGraph):
            graph = graph.to_networkx()

        try:
            return nx.is_directed_acyclic_graph(graph)
        except nx.NetworkXUnfeasible:
            return False

    def reset(self):
        for graph in self._graphs:
            # h_0 = rep_q
            graph.ndata["h"] = graph.ndata["rep"][0].detach().repeat(graph.num_nodes(), 1)
            if "sim" in graph.edata:
                del graph.edata["sim"]

    def to(self, device):
        self._graphs = [graph.to(device) for graph in self._graphs]
        self.device = device

    def __getitem__(self, index: int) -> tuple:
        return self._graphs[index], self.graph_propagator(index, self.num_negatives)
    
    def __len__(self):
        return len(self._graphs)


class TreeHopInferenceDataset(Dataset):
    def __init__(
        self,
        graphs
    ):
        super().__init__()
        self.graphs = graphs

    def __getitem__(self, index: int) -> tuple:
        return self.graphs[index]
    
    def __len__(self):
        return len(self.graphs)
