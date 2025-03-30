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

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["2wiki", "musique", "multihop_rag"],
    )

    # for multihop retrieval
    parser.add_argument(
        "--n_hop", type=int,
        help="Number of hops for multihop retrieval"
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Number of retrieved chunks for each hop"
    )

    # retrieval model and TreeHop settings
    parser.add_argument(
        "--state_dict", type=str,
        help="Resume with saved parameters"
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

    model = get_evaluate_model(args.state_dict)

    df_QA = get_dataset(args.dataset_name)
    retriever = get_retriever(args.dataset_name, model)

    retrieved_result = retriever.multihop_search_passages(
        df_QA["question"].to_list(),
        n_hop=args.n_hop,
        top_n=args.top_n,
        index_batch_size=args.index_batch_size,
        generate_batch_size=args.generate_batch_size,
        redundant_pruning=args.redundant_pruning,
        layerwise_top_pruning=args.layerwise_top_pruning,
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

    # with open(f"eval_data/{args.dataset_name}_result__top_n={args.top_n}&n_hop={args.n_hop}&redundant={args.redundant_pruning}&layerwise_top={args.layerwise_top_pruning}.pkl", "wb") as f:
    #     pickle.dump(retrieved_result, f)

    # with open("eval_data/retrieve_tree_hop.pkl", "rb") as f:
    #     retrieved_passages = pickle.load(f)