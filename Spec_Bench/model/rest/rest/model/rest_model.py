# adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os
import draftretriever



class RestModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        token_spans=[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
    ):
        """
        Args:
            base_model (nn.Module): The LLM to be used.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.token_spans = token_spans

    def get_tokenizer(self):

        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path="codellama/CodeLlama-7b-instruct-hf",
        **kwargs,
    ):
        """
        Args:
            base_model_path (str): Name or path of the LLM to load.

        Returns:
            RestModel
        """
            
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )

        model = cls(
            base_model,
            base_model_path,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass of the LLM.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from the LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])

        if output_orig:
            return outputs, orig
        raise NotImplementedError

    def rest_generate(
        self,
        input_ids,
        datastore,
        temperature=0.0,
        top_p=0.8,
        max_steps=512,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.

        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        self.base_model.model.draft_mask = None

        # Initialize tree attention mask and process prefill tokens
        logits = initialize_logits(
            input_ids, self, past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Retrievd candidates (draft tokens) from the datastore
            candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                logits,
                input_ids,
                datastore,
                self.token_spans,
                device=self.base_model.device
            )
            self.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
            # Use tree attention to verify the candidates and get predictions
            logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                draft_buffers["draft_position_ids"],
                input_ids,
                draft_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, top_p
            )

            # Update the input_ids and logits
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

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


    def baseline_generate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.8,
        max_steps=512,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.

        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        self.base_model.model.draft_mask = None
        outputs = self.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # # Retrievd candidates (draft tokens) from the datastore
            # candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
            #     logits,
            #     input_ids,
            #     datastore,
            #     self.token_spans,
            #     device=self.base_model.device
            # )
            # self.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
            # # Use tree attention to verify the candidates and get predictions
            # logits, outputs = tree_decoding(
            #     self,
            #     tree_candidates,
            #     past_key_values,
            #     draft_buffers["draft_position_ids"],
            #     input_ids,
            #     draft_buffers["retrieve_indices"],
            # )

            # # Evaluate the posterior of the candidates to select the accepted candidate prefix
            # best_candidate, accept_length = evaluate_posterior(
            #     logits, candidates, temperature
            # )

            # # Update the input_ids and logits
            # input_ids, logits, new_token = update_inference_inputs(
            #     input_ids,
            #     candidates,
            #     best_candidate,
            #     accept_length,
            #     draft_buffers["retrieve_indices"],
            #     outputs,
            #     logits,
            #     new_token,
            #     past_key_values_data,
            #     current_length_data,
            # )

            if top_p > 0:
                assert top_p < 1, "top_p should between 0.0 and 1"
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                input_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                input_id = input_id.view(input_id.shape[0], 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values = past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break