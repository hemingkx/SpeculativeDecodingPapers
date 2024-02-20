"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import shortuuid
import torch
import numpy as np
from tqdm import tqdm

from fastchat.utils import str_to_torch_dtype
from fastchat.llm_judge.common import load_questions
from fastchat.model import load_model, get_conversation_template

# Rest imports
import transformers

import sys

sys.path.append("../")

from model.rest.rest.model.utils import *
from model.rest.rest.model.rest_model import RestModel
from model.rest.rest.model.kv_cache import initialize_past_key_values
import draftretriever


def rest_forward(input_ids, model, tokenizer, max_new_token, temperature, top_p, datastore, num_draft, token_spans,
                 max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len + 1
    model.base_model.model.draft_mask = None
    logits = initialize_logits(
        input_ids, model, past_key_values
    )
    new_token = 0

    torch.cuda.synchronize()
    start_time = time.time()
    for idx in range(max_steps):
        candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
            logits,
            input_ids,
            datastore,
            token_spans,
            top_p,
            temperature,
            max_num_draft=num_draft,
            device=model.base_model.device
        )
        model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            draft_buffers["draft_position_ids"],
            input_ids,
            draft_buffers["retrieve_indices"],
        )
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, top_p
        )
        input_ids, logits, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            draft_buffers["retrieve_indices"],
            outputs,
            logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_token:
            break
    return input_ids, new_token, idx, accept_length_list, start_time


def run_eval(
        model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        top_p,
        datastore_path,
        num_draft,
        max_token_span,
        torch_dtype,
):
    questions = load_questions(question_file, question_begin, question_end)

    token_spans = list(range(2, max_token_span + 1))[::-1]
    print("loading the datastore ...")
    datastore = draftretriever.Reader(
        index_file_path=datastore_path,
    )
    print("datastore loaded!")
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                top_p,
                datastore,
                num_draft,
                token_spans,
                torch_dtype,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        top_p,
        datastore,
        num_draft,
        token_spans,
        torch_dtype,
):
    model = RestModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template("vicuna")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # if temperature < 1e-4:
            #     do_sample = False
            # else:
            #     do_sample = True

            # some models may error out when generating long outputs
            try:
                output_ids, new_token, idx, _, start_time = rest_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    max_new_token,
                    temperature,
                    top_p,
                    datastore,
                    num_draft,
                    token_spans,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    accept_lengths_tree = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            accept_lengths_tree_this = []
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # if temperature < 1e-4:
                #     do_sample = False
                # else:
                #     do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids, new_token, idx, accept_length_tree, start_time = rest_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        max_new_token,
                        temperature,
                        top_p,
                        datastore,
                        num_draft,
                        token_spans,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    accept_lengths_tree.extend(accept_length_tree)
                    # if model.config.is_encoder_decoder:
                    #     output_ids = output_ids[0]
                    # else:
                    output_ids = output_ids[0][len(input_ids[0]):]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                accept_lengths_tree_this.extend(accept_length_tree)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths:": accept_lengths_tree_this})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("accept_lengths_tree: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )

    # REST's hyperparameters
    parser.add_argument(
        "--datastore-path",
        type=str,
        required=True,
        help="The path of the datastore for retrival.",
    )

    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The maximum number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0

    args.model_id = args.model_id + "-temperature-" + str(args.temperature) + "-top_p-" + str(args.top_p)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.top_p,
        args.datastore_path,
        args.num_draft,
        args.max_token_span,
        str_to_torch_dtype(args.dtype),
    )

    reorg_answer_file(answer_file)