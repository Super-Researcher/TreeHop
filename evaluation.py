import os
os.environ["DGLBACKEND"] = "pytorch"
import argparse
import torch
import pickle
import functools
import pandas as pd

from src.utils import DEVICE
from passage_retrieval import MultiHopRetriever
from tree_hop.model import TreeHopModel


if DEVICE == "mps":
    DEVICE = "cpu"


def model_file_name_to_params(name):
    lst_params = name.rstrip(".pt").split('__')[1].split('&')
    d_params = dict([param.split('=') for param in lst_params])
    return d_params


@functools.lru_cache()
def get_dataset(dataset_name):
    df_QA = pd.read_json(f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True)
    df_QA = (df_QA[~df_QA["type"].isin(["comparison", # 2wiki
                                        # multihop_rag
                                        "comparison_query", "null_query", "temporal_query"
                                        ])]
             .reset_index())
    df_QA["set_evidence_title"] = df_QA["supporting_facts"].apply(
        lambda lst: set([evd[0] for evd in lst])
    )
    return df_QA


@functools.lru_cache()
def get_evaluate_model(state_dict):
    d_params = model_file_name_to_params(state_dict)
    model = TreeHopModel(
        x_size=1024,
        g_size=int(d_params["g_size"]),
        mlp_size=int(d_params["mlp_size"]),
        n_mlp=int(d_params["n_mlp"]),
        n_head=int(d_params["n_head"])
    )

    pt_state_dict = torch.load(state_dict, weights_only=True, map_location=DEVICE)
    model.load_state_dict(pt_state_dict)
    model.to(DEVICE).compile()
    return model


@functools.lru_cache()
def get_retriever(dataset_name, model):
    retriever = MultiHopRetriever(
        "BAAI/bge-m3",
        passages=f"embedding_data/{dataset_name}/eval_passages.jsonl",
        passage_embeddings=f"embedding_data/{dataset_name}/eval_content_dense.npy",
        tree_hop_model=model,
        projection_size=1024,
        save_or_load_index=True,
        indexing_batch_size=10240,
        index_device=DEVICE
    )
    return retriever


def old_match_retrieve(df):
    set_title = df["set_evidence_title"].copy()
    idx_result = df.name

    lst_match = [0] * len(lst_retrieve)
    for i_hop, retrieved in enumerate(lst_retrieve):
        passages = retrieved[idx_result * top_n**i_hop:
                            idx_result * top_n**i_hop + top_n**i_hop]
        for psg in passages:
            for p in psg:
                if p["title"] not in set_title:
                    continue

                set_title.remove(p["title"])
                lst_match[i_hop] += 1

    return lst_match


def match_retrieve(df, retrieved_passages):
    set_title = df["set_evidence_title"].copy()
    idx_result = df.name

    lst_match = [0] * len(retrieved_passages)
    for i_hop, retrieved in enumerate(retrieved_passages):
        passage = retrieved[idx_result]
        for psg in passage:
            if psg["title"] not in set_title:
                continue

            set_title.remove(psg["title"])
            lst_match[i_hop] += 1

    return lst_match


def evaluate_dataset(
    model,
    dataset_name,
    n_hop,
    top_n,
    index_batch_size=10240,
    generate_batch_size=1024
):
    df_QA = get_dataset(dataset_name)
    lst_questions = df_QA["question"].to_list()

    retriever = get_retriever(dataset_name, model.eval())
    # retriever.search_passages(["Who is "])

    retrieved_result = retriever.multihop_search_passages(
        lst_questions,
        n_hop=n_hop,
        top_n=top_n,
        index_batch_size=index_batch_size,
        generate_batch_size=generate_batch_size,
        return_tree=True
    )

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_result.passage,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    n_total = df_match["set_evidence_title"].map(len).sum()

    return df_match[1].sum(axis=0) / n_total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TreeHop")

    # multihop retrieval with TreeHop
    parser.add_argument(
        "--n_hop", type=int,
        help="Number of hops for multihop retrieval"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["2wiki", "musique", "multihop_rag"],
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Number of retrieved chunks for each hop"
    )

    # TreeHop model and retrieval settings
    parser.add_argument(
        "--state_dict", type=str,
        help="Resume with saved parameters"
    )
    parser.add_argument(
        "--g_size", type=int, default=2048,
        help="Gate size"
    )
    parser.add_argument(
        "--mlp_size", type=int, default=2048,
        help="MLP layer size"
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3,
        help="Number of sequential MLP layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=1,
        help="Number of stacked node modules"
    )
    parser.add_argument(
        "--index_batch_size", type=int, default=10240,
        help="Batch size for Fiass retrieval"
    )
    parser.add_argument(
        "--generate_batch_size", type=float, default=1024,
        help="Batch size for TreeHop inference"
    )
    parser.add_argument(
        "--redundant_pruning", type=bool, default=True,
        help="Toggle stop criterion: redundancy pruning"
    )
    parser.add_argument(
        "--layerwise_top_pruning", type=bool, default=True,
        help="Toggle stop criterion: layer-wise top pruning"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # n_hop = 2
    # top_n = 10
    # redundant_pruning = True
    # layerwise_top_pruning = True
    # index_batch_size=10240
    # generate_batch_size=1024

    # num_negatives = 5
    # x_size = 1024
    # g_size = 2048
    # mlp_size = 2048
    # n_mlp = 3
    # dropout = 0.1
    # n_heads = 1
    # pt_file = "checkpoint/treehop_2wiki=0.123&musique=0.027&multihop_rag=0.093__epoch=9&n_neg=5&neg_mode=paired&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&dropout=0.1&batch_size=64&lr=6e-05&temperature=0.15&weight_decay=2e-08.pt"
    # dataset_name = "2wiki"  # hotpotqa 2wiki musique multihop_rag

    model = get_evaluate_model(args.state_dict)

    # with open("eval_data/retrieve_tree_hop.pkl", "rb") as f:
    #     retrieved_passages = pickle.load(f)
    with open("eval_data/retrieve_graph_reranker.pkl", "rb") as f:
        lst_retrieve = pickle.load(f)


    df_QA = get_dataset(args.dataset_name)
    retriever = get_retriever(args.dataset_name, model)
    # retriever.tree_hop_model = model.eval()
    # res = retriever.search_passages(df_QA["question"].to_list(), top_n=top_n)
    # retrieved_passages = [res[0]]
    retrieved_result = retriever.multihop_search_passages(
        df_QA["question"].to_list(),
        n_hop=args.n_hop,
        top_n=args.top_n,
        # threshold=0.25,
        index_batch_size=args.index_batch_size,
        generate_batch_size=args.generate_batch_size,
        redundant_pruning=args.redundant_pruning,
        layerwise_top_pruning=args.layerwise_top_pruning,
        return_tree=True,
        return_query_similarity=True
    )

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_result.passage,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    # df_match = df_match[~df_match["type"].isin(["comparison", "null_query"])]
    n_total = df_match["set_evidence_title"].map(len).sum()

    print("Recall on each iteration:")
    print(df_match[list(range(args.n_hop))].sum(axis=0) / n_total)

    print("Stats by question type:")
    print(
        df_match.groupby(["type", ])[list(range(args.n_hop))].agg(["count", "mean"])
    )

    k = 0.
    for i, psgs in enumerate(retrieved_result.passage):
        k += sum(map(lambda x: len(x), psgs))
        print(f"Avg. K on hop {i+1}:", k / len(psgs))
