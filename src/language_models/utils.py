import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft.peft_model import PeftModel

from .base import LanguageModel


def assemble_conversation(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages


def ask_model(
    model: LanguageModel,
    tokenizer, 
    prompt: list,
    type="json",
    system_msg: str = None,
    batch_size: int = 4,
    check_if_valid=None
) -> dict:
    parser = get_type_parser(type)
    result = batch_inference(prompt, system_msg, model=model, tokenizer=tokenizer, batch_size=batch_size)

    for res in result:
        try:
            info = parser(res)
        except json.JSONDecodeError:
            return None
        if check_if_valid is not None and not check_if_valid(info):
            return None

    return info


def ask_model_in_parallel(
    model: LanguageModel,
    prompts: list[str],
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid_list: list[Callable] = None,
    max_workers: int = 4,
    desc: str = "Processing...",
    verbose=True,
    mode: Literal["chat", "completion"] = "chat",
):
    if max_workers == -1:
        max_workers = len(prompts)
    assert max_workers >= 1, "max_workers should be greater than or equal to 1"
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if check_if_valid_list is None:
            check_if_valid_list = [None] * len(prompts)
        assert len(prompts) == len(
            check_if_valid_list
        ), "Length of prompts and check_if_valid_list should be the same"
        tasks = {
            executor.submit(
                ask_model, model, prompt, system_msg, type, check_if_valid, mode
            ): idx
            for idx, (prompt, check_if_valid) in enumerate(
                zip(prompts, check_if_valid_list)
            )
        }
        results = []
        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc=desc, disable=not verbose
        ):
            task_id = tasks[future]
            try:
                result = future.result()
                results.append((task_id, result))
            finally:
                ...
        results = [result[1] for result in sorted(results, key=lambda r: r[0])]
        return results


def get_type_parser(type: str) -> Callable:
    def json_parser(result: str):
        # pattern = r"```json(.*?)```"
        pattern = r"{.*?}"
        matches = re.findall(pattern, result, re.DOTALL)
        if matches:
            result = matches[0].strip()
        return json.loads(result)

    def text_parser(result: str):
        return result

    if type == "json":
        return json_parser
    elif type == "text":
        return text_parser
    else:
        raise ValueError(f"Unsupported type: {type}")
    

def model_generate(
    prompt,
    model,
    tokenizer=None,
    temperature=0.8,
    max_new_tokens=1024,
    top_k=1,
    top_p=1.,
    beam_width=1,
    do_sample=False,
    num_return_sequences=1,
    **kwargs
):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if hasattr(model, "device") and model.device.type != "cuda" or isinstance(model, (AutoModel, PeftModel)):
        tokenizer = kwargs.get("tokenizer", tokenizer)
        if tokenizer is None:
            raise ValueError("tokenizer must present if not using cuda")

        inputs = tokenizer(prompt, padding="longest", return_tensors="pt")

        with torch.inference_mode():
            if temperature is None or temperature <= 0.:
                preds = model.generate(
                    **inputs.to(model.device),
                    top_p=top_p,
                    num_beams=beam_width,
                    temperature=None,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **kwargs)
            else:
                preds = model.generate(
                    **inputs.to(model.device),
                    temperature=0. if temperature is None else temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=beam_width,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **kwargs)

        pred_token_ids = preds.sequences[:, inputs.input_ids.shape[1]:]
        pred_text = tokenizer.batch_decode(pred_token_ids)
        pred_log_probs = F.log_softmax(torch.stack(preds.scores), dim=2)
        pred_log_probs = torch.swapaxes(pred_log_probs, 0, 1).to("cpu").numpy()

    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
            use_beam_search=do_sample, n=num_return_sequences, logprobs=5)
        preds = model.generate(prompt, sampling_params, **kwargs)
        pred_token_ids = [[output.token_ids for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_text = [[output.text for output in p.outputs[: num_return_sequences]] for p in preds]
        pred_log_probs = [[output.logprobs for output in p.outputs[: num_return_sequences]] for p in preds]

    return pred_text, pred_token_ids, pred_log_probs



def batch_inference(
        user_query,
        system_msg,
        model,
        tokenizer,
        batch_size=4,
        instruction=None
    ):

    lst_batch = []
    lst_titled_ans = []
    for query in (pbar:= tqdm(user_query, mininterval=1.)):
        messages = assemble_conversation(system_msg, f"{instruction}\n{query}")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        lst_batch.append(prompt)
        if (len(lst_batch) == batch_size) or (len(user_query) - len(lst_titled_ans) == len(lst_batch)):
            lst_output, _, _ = model_generate(
                lst_batch,
                model=model,
                tokenizer=None,
                max_new_tokens=2048,
                temperature=0.,
                top_p=1,
                do_sample=False,
            )

            lst_output = [o[0].rstrip(tokenizer.eos_token).rstrip("<|eot_id|>") for o in lst_output]
            lst_titled_ans.extend(lst_output)

            lst_batch.clear()

    return lst_titled_ans
