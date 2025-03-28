import os
import torch
import transformers

from src import utils
from src.contriever import Contriever, XLMRetriever
from FlagEmbedding import BGEM3FlagModel


def load_retriever(model_path, device="cpu"):
    # try: check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # retriever_model_id = "bert-base-uncased"
            retriever_model_id = "bert-base-multilingual-cased"
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        if "xlm" in retriever_model_id or "xlm" in cfg.model_type:
            model_class = XLMRetriever
        elif "bge-m3" in retriever_model_id or "bge-m3" in cfg.model_type:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        retriever = model_class(cfg)
        pretrained_dict = pretrained_dict["model"]

        if any("encoder_q." in key for key in pretrained_dict.keys()):  # test if model is defined with moco class
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q." in k}
        elif any("encoder." in key for key in pretrained_dict.keys()):  # test if model is defined with inbatch class
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if "encoder." in k}
        retriever.load_state_dict(pretrained_dict, strict=False)
    else:
        retriever_model_id = model_path
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        elif "bge-m3" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(model_class, model_path)

    return retriever, tokenizer, retriever_model_id
