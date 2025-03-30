from uuid import uuid4
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Union
import matplotlib.pyplot as plt

from utils import NodeType


class TreeHopGraph(nx.DiGraph):
    def __init__(
        self,
        query: str,
        passages: List[List[dict]],
        top_n: int,
        redundant_pruning=True,
        layerwise_top_pruning: Union[int, bool] = True,
    ):
        super().__init__()
        self.top_n = top_n
        self.layerwise_top_pruning = top_n if layerwise_top_pruning is True else layerwise_top_pruning

        self._map_uuid = {NodeType.query.value: NodeType.query.value}
        self._duplicate_mask = -5
        self._passage_ids = {self._duplicate_mask}
        self._query_passage_mask: np.array = np.ones((1, 1), dtype=bool)
        self._last_passage_layer = None

        self.add_node(NodeType.query.value, title=f'"{query}"', text='', mask=True)
        self.add_passage_layer(passages,
                               redundant_pruning=redundant_pruning,
                               layerwise_top_pruning=self.layerwise_top_pruning)

    def _id_to_uuid(self, id_):
        uuid = str(uuid4())
        self._map_uuid[uuid] = id_
        return uuid

    def _uuid_to_id(self, uuid):
        return self._map_uuid[uuid]
    
    def _add_passage(self, psg, mask):
        psg_uuid = self._id_to_uuid(psg["id"])
        self.add_node(
            psg_uuid,
            mask=mask,
            id=psg["id"],
            title=f'"{psg["title"]}"',
            text=f'"{psg["text"]}"',
        )
        return psg_uuid

    def _rank_passage_layer(
        self,
        passage_layer,
        redundant_pruning=True,
    ) -> np.array:
        df = pd.DataFrame([psg for passages in passage_layer for psg in passages])
        if redundant_pruning:
            srs_duplicate_mask = df["id"].apply(lambda x: x in self._passage_ids)
        else:
            srs_duplicate_mask = pd.Series([False] * df.shape[0])
        df.loc[srs_duplicate_mask, "score"] = self._duplicate_mask

        idx_rank_score = np.argsort(df["score"].to_numpy())
        idx_rank_score[srs_duplicate_mask] = self._duplicate_mask

        out = idx_rank_score.reshape(len(passage_layer), self.top_n)
        return out

    def add_passage_layer(
        self,
        passage_layer,
        redundant_pruning=True,
        layerwise_top_pruning: Union[int, bool] = True,
        min_ranking=None
    ) -> None:
        last_query_ids = self.get_query_ids()
        assert len(last_query_ids) == len(passage_layer), \
            f"number of new passages must match with number of last query nodes, " \
            f"got {len(last_query_ids)} and {len(passage_layer)}"

        if layerwise_top_pruning is True:
            layerwise_top_pruning = self.top_n

        if layerwise_top_pruning is False:
            self._query_passage_mask = np.ones((len(passage_layer), self.top_n), dtype=bool)
        else:
            layer_ranks = self._rank_passage_layer(
                passage_layer,
                redundant_pruning=redundant_pruning,
            )
            min_ranking = max(layer_ranks.max(axis=None) - layerwise_top_pruning + 1, 0)
            self._query_passage_mask = layer_ranks >= min_ranking

        filtered_passages = []
        for passages, masks, idx_query in zip(passage_layer,
                                              self._query_passage_mask,
                                              last_query_ids):
            for psg, mask in zip(passages, masks):
                if mask and (not redundant_pruning or psg["id"] not in self._passage_ids):
                    filtered_passages.append(psg)
                    self._passage_ids.add(psg["id"])

                psg_uuid = self._add_passage(psg, mask=mask)
                self.add_edge(idx_query, psg_uuid, distance=1. / psg["score"])

            self.nodes[idx_query]["mask"] = False

        self._filtered_passages = filtered_passages

    @property
    def query_passage_mask(self) -> np.array:
        return self._query_passage_mask

    @property
    def filtered_passages(self) -> List[List[dict]]:
        return self._filtered_passages

    def get_query_ids(self):
        return [n for n, v in self.nodes(data=True) if v["mask"]]

    def plot_tree(self, label_attr="id", ax=None):
        # this plot requires pygraphviz package
        pos = nx.nx_agraph.graphviz_layout(self, prog="dot")
        labels = nx.get_node_attributes(self, label_attr)
        edge_labels = nx.get_edge_attributes(self, "distance")
        for key in edge_labels.keys():
            edge_labels[key] = f"{edge_labels[key]:.3f}"

        if ax is None:
            plt.figure(figsize=(25, 10))

        nx.draw(
            self,
            pos,
            arrows=True,
            with_labels=True,
            labels=labels,
            node_size=5,
            node_color=[[0.5, 0.5, 0.5]],
            arrowsize=4,
            ax=ax
        )

        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)
        if ax is None:
            plt.show()
        else:
            ax.show()
    
    def draw_graph(self):
        options = {
            'node_color': 'blue',
            'node_size': 100,
            'width': 3,
            'arrows': True,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(self, **options)
