"""Evidence-chain validation and training-data generation for TreeHop.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import json
import argparse
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from src.normalize_text import normalize
from src.evaluation import normalize_unicode, white_space_fix


# ---------------------------------------------------------------------------
# Graph-topology registry
#
# Mirrors the category sets in tree_hop/dataset.py (SINGLE_ROOT_TYPES,
# COMPARISON_TYPES, BRIDGE_COMPARISON_TYPES, CONVERGENT_TYPES). Kept local so
# this script does not need to import dgl via tree_hop.dataset. If a graph type
# is added there, add it here too.
#
#   structure:
#     "linear"            single entry node, chains to >=1 leaf  (compositional
#                         / inference / bridge / 2hop / 3hop1 / 4hop1)
#     "parallel"          every entry node is also a leaf        (comparison)
#     "bridge_comparison" multiple entry nodes, multiple leaves  (bridge_comparison)
#     "convergent"        multiple entry nodes converge to leaves(3hop2/4hop2/4hop3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Topology:
    structure: str
    n_nodes: int


TOPOLOGY = {
    # single-root chains
    "compositional":     Topology("linear", 2),
    "inference":         Topology("linear", 2),
    "bridge":            Topology("linear", 2),
    "2hop":              Topology("linear", 2),
    "3hop1":             Topology("linear", 3),
    "4hop1":             Topology("linear", 4),
    # comparison (parallel entry == leaf)
    "comparison":        Topology("parallel", 2),
    # bridge comparison
    "bridge_comparison": Topology("bridge_comparison", 4),
    # convergent
    "3hop2":             Topology("convergent", 3),
    "4hop2":             Topology("convergent", 4),
    "4hop3":             Topology("convergent", 4),
}

# Longest-first so "bridge_comparison" wins over "bridge" and "3hop1" over "3hop".
_TYPE_PREFIXES = sorted(TOPOLOGY, key=len, reverse=True)
_ANSWER_SENTINELS = {"", "answer", "none", "null", "leaf", "end", "final"}


def extract_graph_type(sample: dict) -> str | None:
    """Resolve the graph type from an explicit ``type`` field or the id prefix."""
    explicit = sample.get("type")
    if explicit in TOPOLOGY:
        return explicit

    sample_id = str(sample.get("id", ""))
    for prefix in _TYPE_PREFIXES:
        if re.match(rf"{re.escape(prefix)}([^0-9a-zA-Z]|$)", sample_id):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Title cleaning / fuzzy matching (same normalisation as tree_hop/dataset.py)
# ---------------------------------------------------------------------------
def clean_title(title: str) -> str:
    return normalize_unicode(white_space_fix(normalize(str(title)).strip()))


def fuzzy_resolve(title: str, canonical_titles: list[str], cutoff: float = 0.6) -> str | None:
    """Resolve an LLM-produced title to one of the real paragraph titles.

    Exact match first, then cleaned-exact, then substring containment, then a
    difflib close-match fallback --- the appendix's "fuzzy title matching to
    resolve minor discrepancies".
    """
    if title is None:
        return None
    if title in canonical_titles:
        return title

    cleaned = {clean_title(t): t for t in canonical_titles}
    ct = clean_title(title)
    if ct in cleaned:
        return cleaned[ct]

    # substring containment, prefer the most specific (longest) canonical title
    contains = [orig for c, orig in cleaned.items() if c and (c in ct or ct in c)]
    if contains:
        return max(contains, key=len)

    close = difflib.get_close_matches(ct, list(cleaned.keys()), n=1, cutoff=cutoff)
    if close:
        return cleaned[close[0]]
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a meticulous multi-hop reasoning annotator. Given a question, its "
    "answer, and a shuffled set of supporting paragraphs, you determine the "
    "order in which the paragraphs must be read so that each paragraph supplies "
    "the bridge entity needed to reach the next one. You respond with strict "
    "JSON only."
)

_STRUCTURE_HINT = {
    "linear": ("The paragraphs form a single chain: exactly one paragraph is read "
               "first, and each subsequent paragraph depends on the previous one. "
               "Set \"next\" to null for the final paragraph."),
    "parallel": ("The paragraphs are read independently (a comparison): every "
                 "paragraph is an entry point and none depends on another. Set "
                 "\"next\" to null for every paragraph."),
    "convergent": ("Multiple paragraphs are read first and their chains converge "
                   "onto one or more final paragraphs. Set \"next\" to null for "
                   "any paragraph that is final."),
    "bridge_comparison": ("Two independent bridge chains are compared. Each chain "
                          "reads one paragraph then a second that depends on it. "
                          "Set \"next\" to null for the final paragraph of each chain."),
}


def build_prompt(question: str, answer: str, paragraphs: list[dict], topology: Topology) -> str:
    """Structured prompt: question, answer, shuffled supporting paragraphs."""
    para_block = "\n\n".join(
        f"[{i}] Title: {p['title']}\n{p['text']}" for i, p in enumerate(paragraphs)
    )
    titles = ", ".join(json.dumps(p["title"]) for p in paragraphs)
    return (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Supporting paragraphs (in random order):\n{para_block}\n\n"
        f"There are {topology.n_nodes} paragraphs. {_STRUCTURE_HINT[topology.structure]}\n\n"
        "Output a JSON object of the form:\n"
        '{"chain": [{"title": "<paragraph title>", "next": "<title of the paragraph '
        'read next, or null if this paragraph is final>"}, ...]}\n'
        f"Use titles exactly from this set: [{titles}]. "
        "Include every paragraph exactly once. Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# Response parsing + conversion to evidence triplets
# ---------------------------------------------------------------------------
def parse_chain(raw: str) -> list[dict]:
    """Extract the ``chain`` list from a (possibly fenced) JSON response."""
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # tolerate a leading object or a bare array
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    chain = obj["chain"] if isinstance(obj, dict) else obj
    if not isinstance(chain, list):
        raise ValueError("chain is not a list")
    return chain


def chain_to_evidences(
    chain: list[dict], canonical_titles: list[str], answer: str
) -> list[list[str]]:
    """Convert ``{title, next}`` pairs into ``[src, "", dst]`` evidence triplets.

    The relation slot is left empty and the terminal edge points at the answer
    text, matching the answer-leaf convention in tree_hop/dataset.py.
    """
    evidences: list[list[str]] = []
    for link in chain:
        src = fuzzy_resolve(link.get("title"), canonical_titles)
        if src is None:
            raise ValueError(f"unresolved source title: {link.get('title')!r}")

        nxt = link.get("next")
        is_leaf = (
            nxt is None
            or (isinstance(nxt, str) and nxt.strip().lower() in _ANSWER_SENTINELS)
            or (isinstance(nxt, str) and clean_title(nxt) == clean_title(answer))
        )
        if is_leaf:
            evidences.append([src, "", answer])
            continue

        dst = fuzzy_resolve(nxt, canonical_titles)
        if dst is None:
            # a "next" that is neither a real paragraph nor a sentinel -> treat as leaf
            evidences.append([src, "", answer])
        else:
            evidences.append([src, "", dst])
    return evidences


# ---------------------------------------------------------------------------
# Validation (mirrors the assertions in tree_hop/dataset.py::_graph_propagator)
# ---------------------------------------------------------------------------
def validate_evidences(
    evidences: list[list[str]],
    canonical_titles: set[str],
    answer: str,
    topology: Topology,
) -> bool:
    """Acyclic? correct node count? structure matches type? sources in context?"""
    if not evidences:
        return False

    answer_c = clean_title(answer)
    d_nodes: dict[str, list[str]] = {}
    last_layer: set[str] = set()

    for src, _rel, dst in evidences:
        if src not in canonical_titles:          # every source must be a context paragraph
            return False
        d_nodes.setdefault(src, [])
        if clean_title(dst) == answer_c or dst not in canonical_titles or dst == src:
            last_layer.add(src)                  # terminal / answer edge -> leaf
        else:
            if dst not in d_nodes[src]:
                d_nodes[src].append(dst)

    nodes = set(d_nodes) | {d for children in d_nodes.values() for d in children}
    if len(nodes) != topology.n_nodes:
        return False

    g = nx.DiGraph(d_nodes)
    if not nx.is_directed_acyclic_graph(g):
        return False

    start_nodes = set(d_nodes)
    end_nodes = {d for children in d_nodes.values() for d in children}
    first_layer = start_nodes - end_nodes

    s = topology.structure
    if s == "linear":
        return len(first_layer) == 1 and len(last_layer) > 0
    if s == "parallel":
        return len(first_layer) > 1 and first_layer == last_layer
    if s == "bridge_comparison":
        return len(first_layer) > 1 and len(last_layer) > 1
    if s == "convergent":
        return len(first_layer) > 1 and len(last_layer) > 0
    return False


# ---------------------------------------------------------------------------
# Per-sample validation loop (LLM call + retries with rising temperature)
# ---------------------------------------------------------------------------
def supporting_paragraphs(sample: dict) -> list[dict]:
    """The supporting paragraphs, falling back to all ctxs if unannotated."""
    ctxs = sample["ctxs"]
    sf_titles = {clean_title(f[0]) for f in sample.get("supporting_facts", []) if f}
    if not sf_titles:
        return list(ctxs)
    chosen = [c for c in ctxs if clean_title(c["title"]) in sf_titles]
    return chosen or list(ctxs)


def validate_sample(sample: dict, model, temperatures: list[float], max_tokens: int):
    """Return ``sample`` with a validated ``evidences`` field, or None if discarded."""
    gtype = extract_graph_type(sample)
    if gtype is None:
        return None
    topology = TOPOLOGY[gtype]

    paras = supporting_paragraphs(sample)
    canonical = [p["title"] for p in paras]
    canonical_set = set(canonical)
    answer = str(sample.get("answer", ""))

    # Deterministic per-sample shuffle (paper: "supporting paragraphs in random
    # order"); seeded by id so runs are reproducible.
    seed = abs(hash(str(sample.get("id", sample.get("question", ""))))) % (2**32)
    order = list(range(len(paras)))
    import random as _random
    _random.Random(seed).shuffle(order)
    shuffled = [paras[i] for i in order]

    prompt = build_prompt(sample["question"], answer, shuffled, topology)

    for temperature in temperatures:               # rising-temperature retries
        try:
            raw = model.chat(
                prompt, system_msg=SYSTEM_MSG,
                json_mode=True, temperature=temperature, max_tokens=max_tokens,
            )
            chain = parse_chain(raw)
            evidences = chain_to_evidences(chain, canonical, answer)
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
            continue

        if validate_evidences(evidences, canonical_set, answer, topology):
            out = dict(sample)
            out["type"] = gtype
            out["evidences"] = evidences
            return out

    return None


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(provider: str, model_name: str):
    if provider == "deepseek":
        from src.language_models import DeepSeek
        return DeepSeek(model=model_name)
    if provider == "aoai":
        from src.language_models import AOAI
        return AOAI(model=model_name)
    raise ValueError(f"unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evidence-chain validation -> TreeHop train data")
    p.add_argument("--datasets", nargs="+", default=["squad", "nq", "mlqa", "trex"],
                   help="Dataset names; reads train_data/{name}_train_raw.jsonl")
    p.add_argument("--in_dir", default="train_data")
    p.add_argument("--out_dir", default="train_data")
    p.add_argument("--in_suffix", default="_train_raw.jsonl")
    p.add_argument("--out_suffix", default="_train_processed.jsonl")
    p.add_argument("--provider", choices=["deepseek", "aoai"], default="deepseek")
    p.add_argument("--model", default="deepseek-reasoner",
                   help="Reasoning model used to order the evidence chain")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                   help="One LLM attempt per value; rising temperature on retry "
                        "(max 3 retries by default)")
    p.add_argument("--max_tokens", type=int, default=1500)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap samples per dataset")
    return p.parse_args()


def main():
    args = parse_args()
    model = build_model(args.provider, args.model)

    for dataset in args.datasets:
        in_path = os.path.join(args.in_dir, f"{dataset}{args.in_suffix}")
        out_path = os.path.join(args.out_dir, f"{dataset}{args.out_suffix}")
        if not os.path.exists(in_path):
            print(f"[SKIP] {in_path} not found")
            continue

        df = pd.read_json(in_path, lines=True, orient="records")
        samples = df.to_dict(orient="records")
        if args.limit is not None:
            samples = samples[: args.limit]

        validated, discarded = [], 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(validate_sample, s, model, args.temperatures, args.max_tokens): i
                for i, s in enumerate(samples)
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{dataset}: validating chains"):
                result = fut.result()
                if result is None:
                    discarded += 1
                else:
                    validated.append((futures[fut], result))

        # preserve original order
        validated.sort(key=lambda x: x[0])
        rows = [r for _, r in validated]

        os.makedirs(args.out_dir, exist_ok=True)
        pd.DataFrame(rows).to_json(out_path, lines=True, orient="records")
        print(f"{dataset}: kept {len(rows)} / {len(samples)} "
              f"(discarded {discarded}) -> {out_path}")

    print("\nDone. Next: run init_train_vectors.py to vectorize, then train with "
          "training.py --graph_cache_dir train_data/ (graphs + comparison "
          "augmentation are built by tree_hop/dataset.py).")


if __name__ == "__main__":
    main()
