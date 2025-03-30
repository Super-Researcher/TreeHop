# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import time
import glob

import torch
import faiss
import numpy as np
from tqdm.auto import tqdm
from collections import namedtuple
from typing import Union, List, Iterable

import src.data, src.index, src.slurm, src.contriever, src.utils, src.normalize_text
from src.utils import DEVICE
from utils import load_file_jsonl, save_file_jsonl

from tree_hop import TreeHopModel, TreeHopGraph

os.environ["TOKENIZERS_PARALLELISM"] = "true"


search_passage_results = namedtuple(
    "search_passage_results",
    fields := ["passage", "score", "query_embedding", "passage_embedding"],
    defaults=(None,) * len(fields)
)

multihop_search_passage_results = namedtuple(
    "multihop_search_passage_results",
    fields := ["passage", "tree_hop_graph"],
    defaults=(None,) * len(fields)
)


class Retriever:
    def __init__(self,
        model_name_or_path: str,
        passages: str,
        passage_embeddings: str,
        no_fp16=False,
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        per_gpu_batch_size=64,
        query_maxlength=512,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu"
    ):
        self.model_name_or_path = model_name_or_path
        self.passages = passages
        self.passage_embeddings = passage_embeddings
        self.no_fp16 = no_fp16
        self.save_or_load_index = save_or_load_index
        self.indexing_batch_size = indexing_batch_size
        self.lowercase = lowercase
        self.normalize_text = normalize_text
        self.per_gpu_batch_size = per_gpu_batch_size
        self.query_maxlength = query_maxlength
        self.projection_size = projection_size
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.index_device = index_device

        self.setup_retriever()

    @torch.no_grad
    def embed_queries(self, queries):
        embeddings, batch_query = [], []
        for k, q in enumerate(queries):
            if self.lowercase:
                q = q.lower()
            if self.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_query.append(q)

            if len(batch_query) == self.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch_query,
                    return_tensors="pt",
                    max_length=self.query_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(DEVICE) for k, v in encoded_batch.items()}
                output = self.model(**encoded_batch)
                embeddings.append(output.to(self.index_device))

                batch_query.clear()
                # getattr(torch, DEVICE).empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def index_encoded_data(self, input_paths, indexing_batch_size):

        input_paths = glob.glob(self.passage_embeddings)
        input_paths = sorted(input_paths)
        all_ids = []
        all_embeddings = []
        start_idx = 0

        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        for i, file_path in enumerate(input_paths):
            data = src.data.load_regular_data(file_path)
            if isinstance(data, tuple):
                ids, embeddings = data
            else:
                embeddings = data
                ids = list(range(start_idx, start_idx + len(embeddings)))
                start_idx += len(embeddings)

            all_ids.extend(ids)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        while all_embeddings.shape[0] > 0:
            all_embeddings, all_ids = self._batch_add_embeddings(
                all_embeddings, all_ids, indexing_batch_size
            )

        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

    def _batch_add_embeddings(self, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_to_add = ids[:end_idx]
        embeddings_to_add = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        self.indexer.index_data(ids_to_add, embeddings_to_add)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        lst_docs = []
        for passage_ids, scores in top_passages_and_scores:
            lst_doc = []
            for p_id, score in zip(passage_ids, scores):
                doc = passages[p_id].copy()
                doc["score"] = float(score)
                lst_doc.append(doc)

            lst_docs.append(lst_doc)

        return lst_docs

    def setup_retriever(self):
        print(f"Loading model from: {self.model_name_or_path}")
        self.model, self.tokenizer, _ = src.load_retriever(
            self.model_name_or_path, self.index_device
        )
        self.model.eval()
        self.model = self.model.to(DEVICE)
        if not self.no_fp16:
            self.model = self.model.half()

        self.indexer = src.index.Indexer(
            self.projection_size, self.n_subquantizers, self.n_bits
        )
        if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
            if src.slurm.is_distributed():
                self.indexer.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), src.slurm.local_rank, self.indexer.index
                )
            else:
                n_gpus = faiss.get_num_gpus()
                if n_gpus <= 0:
                    raise LookupError("Fiass cannot detect a gpu")

                if n_gpus == 1:
                    self.indexer.index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, self.indexer.index
                    )
                else:
                    self.indexer.index = faiss.index_cpu_to_all_gpus(self.indexer.index)

        # index all passages
        input_paths = glob.glob(self.passage_embeddings)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.save_or_load_index and os.path.exists(index_path):
            self.indexer.deserialize_from(embeddings_dir)
        else:
            self.index_encoded_data(self.passage_embeddings, self.indexing_batch_size)
            if self.save_or_load_index:
                if getattr(self.index_device, "type", self.index_device).startswith("cuda"):
                    self.indexer.index = faiss.index_gpu_to_cpu(self.indexer.index)
                self.indexer.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_regular_data(self.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def get_passage_embedding_by_id(self, passage_ids):
        if isinstance(passage_ids, int):
            return self.indexer.index.reconstruct(passage_ids)

        passage_embedding = []
        for p_id in passage_ids:
            passage_embedding.append(self.indexer.index.reconstruct(p_id))

        return passage_embedding

    def search_passages(
        self,
        query: Union[str, Iterable[str], torch.Tensor, np.ndarray],
        top_n=10,
        index_batch_size=2048,
        return_query_embeddings=False,
        return_passage_embeddings=False
    ):
        queries = [query] if isinstance(query, str) else query

        query_embeddings = \
            queries \
            if isinstance(queries, (torch.Tensor, np.ndarray)) \
            else self.embed_queries(queries)

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()

        # get top k results
        top_ids_and_scores = self.indexer.search_knn(
            query_vectors=query_embeddings,
            top_docs=top_n,
            index_batch_size=index_batch_size
        )

        lst_passages = self.add_passages(self.passage_id_map, top_ids_and_scores)

        lst_passage_embeddings = []
        lst_scores = []
        for passage_ids, scores in top_ids_and_scores:
            lst_scores.append(scores)
            passage_embeddings = self.get_passage_embedding_by_id(passage_ids)
            lst_passage_embeddings.append(passage_embeddings)

        score = np.vstack(lst_scores)
        if return_passage_embeddings:
            passage_embedding = (np.vstack(lst_passage_embeddings)
                                 .reshape((query_embeddings.shape[0], top_n, -1)))
        else:
            passage_embedding = None

        return search_passage_results(
            passage=lst_passages,
            score=score,
            query_embedding=query_embeddings if return_query_embeddings else None,
            passage_embedding=passage_embedding
        )


class MultiHopRetriever(Retriever):
    def __init__(
        self,
        model_name_or_path,
        passages,
        passage_embeddings,
        tree_hop_model: TreeHopModel,
        no_fp16=True,
        save_or_load_index=False,
        indexing_batch_size=1000000,
        lowercase=False,
        normalize_text=True,
        per_gpu_batch_size=64,
        query_maxlength=512,
        projection_size=768,
        n_subquantizers=0,
        n_bits=8,
        index_device="cpu"
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            passages=passages,
            passage_embeddings=passage_embeddings,
            no_fp16=no_fp16,
            save_or_load_index=save_or_load_index,
            indexing_batch_size=indexing_batch_size,
            lowercase=lowercase,
            normalize_text=normalize_text,
            per_gpu_batch_size=per_gpu_batch_size,
            query_maxlength=query_maxlength,
            projection_size=projection_size,
            n_subquantizers=n_subquantizers,
            n_bits=n_bits,
            index_device=index_device
        )

        self.tree_hop_model = tree_hop_model

    def multihop_search_passages(
        self,
        query: Union[List[str], str],
        n_hop: int,
        top_n=10,
        index_batch_size=10240,
        generate_batch_size=1024,
        show_progress=True,
        redundant_pruning=True,
        layerwise_top_pruning: Union[int, bool] = True,
        return_tree=False
    ):
        assert isinstance(n_hop, int) and n_hop > 0, "n_hop must be a positive integer"
        pbar = tqdm(
            total=n_hop,
            desc="Retrieving",
            postfix={"num_query": len(query)},
            leave=True,
            disable=not show_progress
        )

        query = [query] if isinstance(query, str) else query
        # start_time_search = time.time()
        search_result = self.search_passages(
            query,
            top_n=top_n,
            index_batch_size=index_batch_size,
            return_query_embeddings=True,
            return_passage_embeddings=True
        )
        # pbar.set_postfix({"num_query": len(query_embeddings),
        #                   "elapsed": time.time() - start_time_search})

        self.tree_hop_model.reset_query()

        pbar.set_description("Generating")
        # start_time_generate = time.time()
        q_emb = self.tree_hop_model.next_query(
            q_emb=search_result.query_embedding,
            ctx_embs=search_result.passage_embedding,
            batch_size=generate_batch_size
        )
        pbar.set_postfix({"num_query": len(q_emb),
                        #   "elapsed": time.time() - start_time_generate
                          })

        tree_hop_graphs = [TreeHopGraph(q, [psg], top_n=top_n,
                                        redundant_pruning=redundant_pruning,
                                        layerwise_top_pruning=layerwise_top_pruning)
                           for q, psg in zip(query, search_result.passage)]

        lst_results = [[graph.filtered_passages for graph in tree_hop_graphs]]
        pbar.update(1)

        if n_hop == 1:
            pbar.close()
            return multihop_search_passage_results(
                passage=lst_results,
                tree_hop_graph=tree_hop_graphs if return_tree else None,
                # query_similarity=query_sims if return_query_similarity else None
            )

        query_passage_masks = [graph.query_passage_mask for graph in tree_hop_graphs]
        ary_query_passage_masks = np.concatenate(query_passage_masks, axis=None)
        last_q_emb = q_emb[ary_query_passage_masks]

        for i_hop in range(1, n_hop):
            pbar.set_description("Retrieving")
            # start_time_search = time.time()
            search_result = self.search_passages(
                last_q_emb,
                top_n=top_n,
                index_batch_size=index_batch_size,
                return_passage_embeddings=True
            )

            pbar.set_description("Generating")
            query_passage_masks = [graph.query_passage_mask for graph in tree_hop_graphs]
            # start_time_generate = time.time()
            # assume embeddings reconstructed from faiss are normalized before stored
            # filter out semantically distant passage embedddings
            q_emb = self.tree_hop_model.next_query(
                ctx_embs=search_result.passage_embedding.reshape(-1, last_q_emb.shape[1]),
                query_passage_masks=query_passage_masks,
                batch_size=generate_batch_size
            )
            pbar.set_postfix({"num_query": len(q_emb),
                            #   "elapsed": time.time() - start_time_generate
                              })

            query_passage_masks = []
            lst_passages = []
            i_current = 0
            for i, graph in enumerate(tree_hop_graphs):
                num_query = graph.query_passage_mask.sum(axis=None)
                if num_query <= 0:
                    lst_passages.append([])
                    continue

                passage_layer = search_result.passage[i_current: i_current + num_query]
                graph.add_passage_layer(
                    passage_layer,
                    redundant_pruning=redundant_pruning,
                    layerwise_top_pruning=layerwise_top_pruning,
                )

                query_passage_masks.append(graph.query_passage_mask)
                lst_passages.append(graph.filtered_passages)
                i_current += num_query

            ary_query_passage_masks = np.concatenate(query_passage_masks, axis=None)
            last_q_emb = q_emb[ary_query_passage_masks]

            lst_results.append(lst_passages)
            pbar.update(1)
            pbar.set_postfix({"num_query": len(last_q_emb),
                            #   "elapsed": time.time() - start_time_search
                              })

        pbar.close()
        return multihop_search_passage_results(
            passage=lst_results,
            tree_hop_graph=tree_hop_graphs if return_tree else None
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passage_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--output", type=str, help="dir path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards > 1:
        src.slurm.init_distributed_mode(args)

    # for debugging
    # data_paths = glob.glob(args.data)
    retriever = Retriever(
        model_name_or_path=args.model_name_or_path,
        passages=args.passages,
        passage_embeddings=args.passage_embeddings,
        no_fp16=args.no_fp16,
        save_or_load_index=args.save_or_load_index,
        indexing_batch_size=args.indexing_batch_size,
        lowercase=args.lowercase,
        normalize_text=args.normalize_text,
        per_gpu_batch_size=args.per_gpu_batch_size,
        query_maxlength=args.question_maxlength,
        projection_size=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits
    )

    query = args.query
    if os.path.exists(query):
        query = load_file_jsonl(query)

        shard_size = len(query) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size
        if args.shard_id == args.num_shards - 1:
            end_idx = len(query)

        query = query[start_idx: end_idx]
        print("query length:", len(query))

    retrieved_documents = retriever.search_passages(query, args.n_docs).passage
    if isinstance(args.output, str):
        if isinstance(query, str):
            data = [{"question": query, "ctxs": retrieved_documents}]
        else:
            data = [{"question": question, "ctxs": ctx}
                    for question, ctx in zip(query, retrieved_documents)]
        save_file_jsonl(data, args.output)
    else:
        print(retrieved_documents)


if __name__ == "__main__":
    # --query "What is the occupation of Obama?" --passages ./wikipedia_data/psgs_w100.tsv --passage_embeddings "./wikipedia_data/embedding_contriever-msmarco/*" --model_name_or_path "facebook/contriever-msmarco" --output ./train_data/extractor_retrieve_wiki.jsonl
    # main()
    n_hop=2
    top_n=10

    num_negatives = 5
    x_size = 1024
    g_size = 2048
    mlp_size = 2048
    n_mlp = 3
    dropout = 0.1
    n_heads = 1
    pt_file = "checkpoint/checkpoint/treehop_2wiki=0.056&musique=0.019&multihop_rag=0.098__epoch=13&n_neg=5&neg_mode=paired&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&dropout=0.1&batch_size=64&lr=6e-05&temperature=0.15&weight_decay=2e-08&seed=1307.pt"

    graph_reranker = TreeHopModel(x_size, g_size, mlp_size, n_mlp, dropout, n_heads)
    pt_state_dict = torch.load(pt_file, weights_only=True, map_location=DEVICE)
    graph_reranker.load_state_dict(pt_state_dict)
    graph_reranker.eval().to(DEVICE).compile()

    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages="embedding_data/multihop_rag/eval_passages.jsonl",
        passage_embeddings="embedding_data/multihop_rag/eval_content_dense.npy",
        tree_hop_model=graph_reranker,
        projection_size=1024,
        save_or_load_index=True,
        indexing_batch_size=10240,
        index_device="cuda"
    )
    import pandas as pd
    df_QA = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/MultiHopRAG.json")
    df_QA = df_QA.loc[df_QA["question_type"] != "null_query", :]
    lst_questions = df_QA.loc[df_QA["question_type"] != "null_query", "query"].to_list()

    retrieve_result = retriever.multihop_search_passages(
        lst_questions,
        n_hop=n_hop,
        top_n=top_n,
        index_batch_size=2048
    )
    print(retrieve_result)

    # python passage_retrieval.py