# TreeHop: Generate and Filter Next Query Embeddings Efficiently for Multi-hop Question Answering

## System Requirement
> Ubuntu 18.06 LTS+ or MacOS Sequoia+.
  Nvidia GPU with 32GB of RAM at minimum.
  16GB of system RAM for [reproduction](#reproduction), 64GB for [training](#train-treehop).
  50GB of free space on hard drive.


## Python Environment
Please refer to [requirements.txt](/requirements.txt)

### Embedding Preliminary
The repository comes with evaluate embedding database, activate git lfs to pull the data:
```sh
git lfs pull
```

For full embedding database generation, run the following two scripts that generate training and evaluate embedding database.
```sh
python init_train_vectors.py
python init_multihop_rag.py
```

## Reproduction
### To evaluate TreeHop multihop retrieval, run the following code. Here we take 2WikiMultihop dataset and recall@5 with three hops as example.
* To change dataset, replace `2wiki` with `musique` or `multihop_rag`.
* Revise `n_hop` and `top_n` to change number of hops and top retrieval settings. 
* Toggle `redundant_pruning` and `layerwise_top_pruning` to reproduce our ablation study on stop criterion.

```sh
python evaluation.py \
    --state_dict "checkpoint/treehop__epoch=8&n_neg=5&neg_mode=paired&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&dropout=0.1&batch_size=64&lr=6e-05&temperature=0.15&weight_decay=2e-08.pt" \
    --dataset_name multihop_rag \
    --n_hop 3 \
    --top_n 5 \
    --redundant_pruning True \
    --layerwise_top_pruning True
```


## Train TreeHop
Run the following code to generate graph and train TreeHop. Please refer to `parse_args` function in the `training.py` for arguments to this script.
```python
python training.py --graph_cache_dir ./train_data/
```
