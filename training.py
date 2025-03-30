import os
os.environ["DGLBACKEND"] = "pytorch"
import argparse
from collections import namedtuple
from tqdm.auto import tqdm

import random
import numpy as np
import dgl
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from info_nce import info_nce

from src.utils import DEVICE
from utils import NodeType
from tree_hop import TreeHopModel, TreeHopTrainDataset
from evaluation import evaluate_dataset

try:
   mp.set_start_method('spawn')
   print("Multiprocess already spawned")
except RuntimeError:
   pass


InBatch = namedtuple(
    "Graph_Reranker_Batch",
    ["graph", "prop"]
)


def seed_env(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE != "cpu":
        getattr(torch, DEVICE).manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**3
    seed_env(worker_seed)


def gather_graph_contrastive_losses(
    g: dgl.DGLGraph,
    num_negatives,
    negative_mode="paired",
    temperature=0.1
):
    y = g.ndata["y"]
    b_positives = (y == NodeType.relevant_doc.value) | (y == NodeType.leaf.value)
    b_negatives = y == NodeType.irrelevant_doc.value

    idx_out_nodes, idx_in_nodes = g.edges()
    idx_positives = b_positives.nonzero().flatten()
    idx_query_nodes = idx_out_nodes[torch.isin(idx_in_nodes, idx_positives)]

    queries = g.ndata["h"][idx_query_nodes]
    positives = g.ndata["rep"][b_positives]
    negatives = g.ndata["rep"][b_negatives]
    if negative_mode == "paired":
        negatives = negatives.view(positives.size()[0], -1, positives.size()[1])
        if negatives.shape[1] < num_negatives:
            raise LookupError("number of negatives less than specified")

        idx_picked_negs = torch.randint(negatives.shape[1], (num_negatives,), device=g.device)
        negatives = negatives[:, idx_picked_negs, :]
    elif negative_mode == "unpaired":
        idx_picked_negs = torch.randint(negatives.shape[0], (num_negatives,), device=g.device)
        negatives = negatives[idx_picked_negs, :]
    else:
        raise NotImplementedError()

    # lst_idx_queries, lst_idx_positives, lst_idx_negatives = [], [], []
    # gen = list(dgl.topological_nodes_generator(g))
    # # skip roots as they do not have gradients
    # current_idxs = gen[1].int().to(g.device)
    # # skip negative nodes as they don't have successors
    # current_idxs = current_idxs[b_positives[current_idxs]]
    # while len(current_idxs) > 0:
    #     lst_tmp = []
    #     for node in current_idxs:
    #         successors = g.successors(node)
    #         if len(successors) == 0:
    #             # is leaf node
    #             continue

    #         query = g.ndata["h"][node]
    #         positives = g.ndata["rep"][successors][b_positives[successors]]
    #         negatives = g.ndata["rep"][successors][b_negatives[successors]]
    #         negatives = negatives.view(positives.size()[0], -1, positives.size()[1])

    #         for pos, negs in zip(positives, negatives):
    #             lst_idx_queries.append(query)
    #             lst_idx_positives.append(pos)
    #             lst_idx_negatives.append(negs)

    #         lst_tmp.extend(successors[b_positives[successors]])

    #     current_idxs = lst_tmp

    # loss = info_nce(
    #     query=torch.stack(lst_idx_queries, dim=0),
    #     positive_key=torch.stack(lst_idx_positives, dim=0),
    #     negative_keys=torch.stack(lst_idx_negatives, dim=0),
    #     negative_mode='paired',
    #     temperature=temperature
    # )
    loss = info_nce(
        query=queries,
        positive_key=positives,
        negative_keys=negatives,
        negative_mode=negative_mode,
        temperature=temperature
    )
    return loss


def collate_batch(batch):
    batch_graph, batch_props = zip(*batch)
    return InBatch(
        graph=dgl.batch(batch_graph).to(DEVICE),
        prop=batch_props
    )


def evaluate_retrieve(
    model,
    n_hop,
    top_n,
    index_batch_size=10240,
    generate_batch_size=512
):
    datasets = ["2wiki", "musique", "multihop_rag"]
    d_stats = {}
    for dataset_name in datasets:
        stat = evaluate_dataset(
            model,
            dataset_name,
            n_hop,
            top_n,
            index_batch_size=index_batch_size,
            generate_batch_size=generate_batch_size
        )
        d_stats[dataset_name] = stat

    return d_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeHop")

    parser.add_argument(
        "--graph_cache_dir", type=str, default=None,
        help="Load dgl graph cache and respective dataset"
    )
    parser.add_argument(
        "--state_dict", type=str, default=None,
        help="Resume with saved parameters"
    )
    parser.add_argument(
        "--n_neg", type=int, default=5,
        help="Number of negatives for each positive sample for contrastive learning"
    )
    parser.add_argument(
        "--neg_mode", type=str, default="paired",
        choices=['paired', 'unpaired'],
        help="Type of negatives w.r.t query for Info NCE loss"
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
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--epoch", type=int, default=20,
        help="Number of training epoch"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Number of training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=6e-5,
        help="Training learning rate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.15,
        help="InfoNCE temperature"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=2e-8,
        help="Training weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # hyper parameters
    args = parse_args()

    if DEVICE == "mps":
        # dgl does not support mps on Metal
        DEVICE = "cpu"

    print(f"Using {DEVICE}")

    if isinstance(args.seed, int):
        seed_env(args.seed)

    # create datasets
    train_set = TreeHopTrainDataset(
        "2wiki", "train",
        num_negatives=args.n_neg,
        # negative_dataset="embedding_data/hotpotqa/train_dense.npy",
        graph_cache_dir=args.graph_cache_dir,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=0,
    )

    # create the model
    model = TreeHopModel(
        x_size=train_set.ary_contexts.shape[-1],
        g_size=args.g_size,
        mlp_size=args.mlp_size,
        n_mlp=args.n_mlp,
        dropout=args.dropout,
        n_head=args.n_head
    )
    if args.state_dict is not None:
        pt_state_dict = torch.load(args.state_dict, weights_only=True, map_location=DEVICE)
        model.load_state_dict(pt_state_dict)
        print(f"Model checkpoint '{args.state_dict}' loaded")

    print(model.to(DEVICE))

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # train_loader = eval_loader
    # training loop
    epoch_train_loss = []
    epoch_eval_loss = []
    epoch_eval_score = []
    for epoch in (epoch_pbar:=tqdm(range(args.epoch),
                                   desc="Epoch",
                                   position=0,
                                   leave=True)):
        model.train()
        for step, batch in enumerate(pbar:=tqdm(train_loader,
                                                desc="Train step",
                                                position=1,
                                                leave=True,
                                                mininterval=1.)):
            g = batch.graph
            optimizer.zero_grad()
            h = model(g)
            loss = gather_graph_contrastive_losses(
                g, negative_mode=args.neg_mode, num_negatives=args.n_neg, temperature=args.temperature
            )
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3f}"}, refresh=False)

        d_stats = evaluate_retrieve(model.eval(), n_hop=2, top_n=5)
        epoch_pbar.set_postfix(d_stats)

        str_eval_stats = '&'.join([f'{k}={v:.3f}' for k, v in d_stats.items()])
        str_args = '&'.join([f'{k}={v}'
                             for k, v in args.__dict__.items()
                             if k not in ("epoch", "graph_cache_dir", "state_dict")
                             and v is not None])
        torch.save(
            model.state_dict(),
            f"checkpoint/treehop_{str_eval_stats}__epoch={epoch}&{str_args}.pt",
        )

        train_loader.dataset.reset()
        epoch_train_loss.clear()
        epoch_eval_loss.clear()
        epoch_eval_score.clear()

    # python training.py --graph_cache_dir checkpoint/