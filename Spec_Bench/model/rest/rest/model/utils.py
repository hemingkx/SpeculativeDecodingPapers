# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

import numpy as np
import torch
import torch.nn.functional as F
import draftretriever

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))



def initialize_logits(input_ids, model, past_key_values):
    """
    Forward pass through the model to obtain the model outputs, and logits.


    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model: The LLM for generation.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - logits (torch.Tensor): logits from the LLM.
    """
    outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    return logits


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates_and_draft_buffer(logits, input_ids, datastore, token_spans, top_p=0., temperature=1., max_num_draft=64, device="cuda"):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - logits (torch.Tensor): Original logits.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    
    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """

    # Greedy decoding: Select the most probable candidate from the original logits.
    if top_p == 0:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        assert top_p < 1, "top_p should between 0.0 and 1"
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
        candidates_logit = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(0)

    input_ids_extend = torch.cat([input_ids.squeeze(0), candidates_logit], dim=-1)
        
    retrieved_token_list = []
    _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = [], [], [], []
    for span_id, token_span in enumerate(token_spans):
        this_token = input_ids_extend.squeeze(0)[-token_span:].to("cpu").tolist()
        # Retrieve draft tokens from the datastore, and get draft buffer
        retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(this_token, choices=max_num_draft)
    
        # No retrieved sequences
        if len(retrieved_token_list) == 0:
            continue
        # Break because this span has hitted
        else:
            break
    # TODO: just continue to the next retrieval process
    if len(retrieved_token_list) == 0:
        # Just randomlt guess one token
        random_index = 100
        retrieved_position_token_list = [[random_index]]
        _draft_attn_mask = [[1., 0.], [1., 1.]]
        _tree_indices = [0, 1]
        _draft_position_ids = [0, 1]
        _retrieve_indices = [[0, 1]]
    else:
        retrieved_position_token_list = [list(row) for row in zip(*retrieved_token_list)]
        retrieved_position_token_list = [[x for i, x in enumerate(sublist) if sublist.index(x) == i and x != -2] for sublist in retrieved_position_token_list]
        TOPK = max(len(retrieved_position_token) for retrieved_position_token in retrieved_position_token_list)
        retrieved_position_token_list = [pad_path(retrieved_position_token, TOPK) for retrieved_position_token in retrieved_position_token_list]
        
    # Aggregate the generated buffers into a dictionary and Move the tensors in the dictionary to the specified device
    draft_buffers = {
        "draft_attn_mask": torch.tensor(_draft_attn_mask, device=device).unsqueeze(0).unsqueeze(0),
        "tree_indices": torch.tensor(_tree_indices, device=device),
        "draft_position_ids": torch.tensor(_draft_position_ids, device=device),
        "retrieve_indices": torch.tensor(_retrieve_indices, device=device),
        }
    
    candidates_draft_logits = torch.tensor(retrieved_position_token_list, dtype=torch.long, device=candidates_logit.device).contiguous()

    # Combine the selected candidate from the original logits with the draft logits.
    candidates = torch.cat([candidates_logit, candidates_draft_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[draft_buffers["tree_indices"]]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[draft_buffers["retrieve_indices"]]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates, draft_buffers


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    draft_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - draft_position_ids (torch.Tensor): Positional IDs (Layer IDs in the Trie) of each draft token.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the draft position IDs to the length of the input sequence.
    position_ids = draft_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return each draft token's logits, and possibly other outputs.
    outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]

    return logits, outputs

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):

    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature

    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)

    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)


    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')

    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def evaluate_posterior(
    logits, candidates, temperature, top_p=0.8
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    elif top_p > 0:
        assert top_p < 1.0, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]

    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token


def top_p_filtering(logits, top_p=0.0, filter_value=float('-inf')):
    # from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79


    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits