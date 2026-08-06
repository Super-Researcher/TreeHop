# TreeHop: Efficient Embedding-Level Query Rewriter 

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
* To change dataset, replace `2wiki` with `musique`, `multihop_rag` or `hotpotqa_distractor`.
* Revise `n_hop` and `top_n` to change number of hops and top retrieval settings. 
* Toggle `prune_redundant` and `prune_layer_top` to reproduce our ablation study on stop criterion.

```sh
python evaluation.py \
    --model_name_or_path "checkpoint/infonce_hotpotqa=0.055&musique=0.051&multihop=0.093__epoch=2&loss=infonce&n_neg=6&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_layer=3&norm=rms&lr=2e-05&seed=1307.pt" \
    --dataset_name multihop_rag \
    --n_hop 3 \
    --top_n 5 \
    --prune_redundant \
    --prune_layer_top
```


## Train TreeHop
Run the following code to generate graph and train TreeHop. Please refer to `parse_args` function in the `training.py` for arguments to this script.
```python
python training.py --graph_cache_dir ./train_data/
```
