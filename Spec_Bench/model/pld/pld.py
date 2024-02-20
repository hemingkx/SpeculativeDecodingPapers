import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import _crop_past_key_values

device = torch.device('cuda:0')

@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)


@torch.no_grad()
def greedy_search_pld(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        draft_matching_window_size=3,
        draft_num_candidate_tokens=10,
        **model_kwargs,
):
    global tokenizer

    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    i = 0

    while True:
        i += 1
        cur_len = input_ids.shape[-1]

        candidate_pred_tokens = find_candidate_pred_tokens(input_ids, draft_matching_window_size,
                                                           draft_num_candidate_tokens)

        if len(candidate_pred_tokens) == 0:
            candidate_pred_tokens = torch.tensor([100], device=input_ids.device).unsqueeze(0)
        else:
            candidate_pred_tokens = candidate_pred_tokens.unsqueeze(0)

        candidate_input_ids = torch.cat((input_ids, candidate_pred_tokens), dim=1)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

        candidate_kwargs = copy.copy(model_kwargs)

        attention_mask = candidate_kwargs["attention_mask"]
        mask_extension_length = candidate_input_ids.shape[1] - attention_mask.shape[1]
        candidate_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], mask_extension_length))], dim=-1,)

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        new_logits = outputs.logits[:, -candidate_length - 1:]  # excludes the input prompt if present
        selected_tokens = new_logits.argmax(dim=-1)
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

        n_matches = min(n_matches, max_len - cur_len - 1)

        valid_tokens = selected_tokens[:, : n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        new_cur_len = input_ids.shape[-1]

        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        model_kwargs["past_key_values"] = outputs.past_key_values

        # stop if we exceed the maximum length

        if (valid_tokens == eos_token_id_tensor.item()).any():
            break

        if stopping_criteria(input_ids, scores):
            break

    return input_ids