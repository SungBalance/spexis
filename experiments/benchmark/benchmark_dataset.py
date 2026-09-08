# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from the
datasets used by the Spexis experiments. Each dataset subclass of
BenchmarkDataset implements sample generation. Supported dataset types:
  - SpecBench
  - ShareGPT
  - UltraChat
  - LMSYS-Chat
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from conversation import Conversation, SeparatorStyle

from spexis.multimodal import MultiModalDataDict
from spexis.transformers_utils.tokenizer import AnyTokenizer

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def apply_multimodal_chat_transformation(
            self,
            prompt: str,
            mm_content: Optional[MultiModalDataDict] = None) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")


    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = (len(completion_ids)
                              if output_len is None else output_len)
            if not is_valid_sequence(prompt_len,
                                     new_output_len,
                                     skip_min_output_len_check=output_len
                                     is not None):
                continue
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, None)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples
    
# -----------------------------------------------------------------------------
# SpecBench Dataset Implementation
# -----------------------------------------------------------------------------


class SpecBenchDataset(BenchmarkDataset):
    """
    Implements the SpecBench dataset.  Loads data from a jsonl file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
        
        self.data = []
        with open(self.dataset_path, "r") as ques_file:
            for line in ques_file:
                if line:
                    self.data.append(json.loads(line))
        
        # NOTE: the system message must match the one used to train the
        # exit models.
        self.conversation_template = Conversation(
                                        name="vicuna_v1.1",
                                        system_message="A chat between a curious user and an artificial intelligence assistant. "
                                        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
                                        roles=("USER", "ASSISTANT"),
                                        sep_style=SeparatorStyle.ADD_COLON_TWO,
                                        sep=" ",
                                        sep2="</s>",
                                    )
        
        # Filter entries with only one conversation turn. -> reduces questions to 400 from 480
        # self.data = [
        #     entry for entry in self.data
        #     if "turns" in entry and len(entry["turns"]) == 1
        # ]

        #for idx, entry in enumerate(self.data):
        #    if len(entry["turns"]) > 1:
        #        new_turns = entry["turns"][0]
        #        self.data[idx]["turns"] = new_turns


        random.seed(self.random_seed)
        random.shuffle(self.data)

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            total_req_num = len(requests) 
            orig_requests = requests[:]
            copy_time = num_requests // total_req_num
            remain = num_requests % total_req_num
            for i in range(copy_time):
                if (i == copy_time - 1) and (remain != 0):
                    requests.extend(orig_requests[:remain])
                else:
                    requests.extend(orig_requests)
            
            assert len(requests) == num_requests, f"should assert # request to {num_requests}, but only {len(requests)} was ready"



    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            # prompt, completion = (
            #     entry["conversations"][0]["value"],
            #     entry["conversations"][1]["value"],
            # )

            if not (kwargs["task"] == entry["category"]
                    or kwargs["task"] == "total"):
                continue

            # qs = entry["turns"][0]
            # self.conversation = self.conversation_template.copy()

            # self.conversation.append_message(
            #     self.conversation.roles[0], qs)
            # self.conversation.append_message(
            #     self.conversation.roles[1], None)
            # self.conversation.stop_str = "</s>"
            # prompt = self.conversation.get_prompt()

            conversation = [{"role": "system", "content": "You are a helpful assistant."}]
            conversation.extend([
                {"role": "user", "content": entry['turns'][0]},
                ])
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)

            prompt_ids = tokenizer(prompt).input_ids
            # completion_ids = tokenizer(completion).input_ids
            
            prompt_len = len(prompt_ids)
            max_output_len = output_len #kwargs["output_len"]
            assert max_output_len == 2048, "Please provide output_len for SpecBench dataset."
            

            if not is_valid_sequence(prompt_len,
                                     max_output_len,
                                     max_total_len=4096,
                                     skip_min_output_len_check=(max_output_len
                                                                is not None)):
                continue

            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=max_output_len,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class UltraChatDataset(BenchmarkDataset):
    """
    Implements the UltraChat dataset. Loads data from a Parquet file and generates
    sample requests based on multi-turn conversations. The prompt includes all
    conversation turns up to (but not including) the last assistant response.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # Load parquet file using pandas
        self.data = pd.read_parquet(self.dataset_path)

        # Filter entries that have messages column and at least one message
        if "messages" not in self.data.columns:
            raise ValueError("Dataset must have a 'messages' column.")
        
        # Filter entries with at least one message
        self.data = self.data[self.data["messages"].notna()]
        
        # Convert messages to list if it's stored as JSON string or other types
        def parse_messages(x):
            # Handle None/NaN
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            
            # Handle string (JSON)
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return None
            
            # Handle numpy array (common in parquet files)
            if hasattr(x, '__iter__') and not isinstance(x, (str, dict)):
                try:
                    # Convert numpy array, PyArrow ListArray, or other iterables to list
                    if hasattr(x, 'tolist'):
                        result = x.tolist()
                    elif hasattr(x, 'to_pylist'):
                        result = x.to_pylist()
                    else:
                        result = list(x)
                        print("this is list")

                    # Validate that result is a list of dicts
                    if isinstance(result, list) and len(result) > 0:
                        # Check if first element is a dict (valid message format)
                        if isinstance(result[0], dict):
                            return result
                        # If it's a list of lists, flatten it
                        if isinstance(result[0], list):
                            return [item for sublist in result for item in sublist if isinstance(item, dict)]
                    return result
                except (TypeError, AttributeError, ValueError):
                    return None
            
            # Handle list-like types (list, tuple, etc.)
            if isinstance(x, (list, tuple)):
                return list(x)
            
            # Handle dict (single message wrapped in dict)
            if isinstance(x, dict):
                return [x] if "role" in x and "content" in x else None
            
            # Return as-is if already a list
            return x
        
        self.data["messages"] = self.data["messages"].apply(parse_messages)
        
        # Filter: keep only entries with valid list of messages
        def is_valid_messages(x):
            if x is None:
                return False
            if not isinstance(x, list):
                return False
            if len(x) == 0:
                return False
            # Check if all items are dicts with required keys
            return all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in x)
        
        self.data = self.data[self.data["messages"].apply(is_valid_messages)]
        
        # Convert to list of dicts for easier iteration
        self.data = self.data.to_dict("records")
        
        # Shuffle the list after conversion
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            total_req_num = len(requests) 
            orig_requests = requests[:]
            copy_time = num_requests // total_req_num
            remain = num_requests % total_req_num
            for i in range(copy_time):
                if (i == copy_time - 1) and (remain != 0):
                    requests.extend(orig_requests[:remain])
                else:
                    requests.extend(orig_requests)
            
            assert len(requests) == num_requests, f"should assert # request to {num_requests}, but only {len(requests)} was ready"

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break

            messages = entry["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                continue
            
            # Validate message format
            if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
                continue
            
            # Find the last assistant message index
            last_assistant_idx = len(messages) - 1
            #for i in range(len(messages) - 1, -1, -1):
            #    if messages[i].get("role") == "assistant":
            #        last_assistant_idx = i
            #        break
            
            # If no assistant message found, skip this entry
            if last_assistant_idx == -1:
                continue
            
            # Get all messages up to (but not including) the last assistant message
            prompt_messages = messages[:last_assistant_idx]
            
            # Skip if prompt_messages is empty (last assistant is the first message)
            if len(prompt_messages) == 0:
                continue
            
            # Convert to chat format using tokenizer's apply_chat_template
            # If there's no system message, add a default one
            if prompt_messages[0].get("role") != "system":
                conversation = [{"role": "system", "content": "You are a helpful assistant."}]
                conversation.extend(prompt_messages)
            else:
                conversation = prompt_messages
            
            # Apply chat template to create the prompt
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            prompt_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_ids)
            
            # Use provided output_len or a default value
            new_output_len = output_len if output_len is not None else 512
            
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                max_total_len=4096,
                max_prompt_len=2048,                
                skip_min_output_len_check=output_len is not None
            ):
                continue
                        
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                )
            )
        self.maybe_oversample_requests(samples, num_requests)
        return samples

# -----------------------------------------------------------------------------
# lmsys-chat-1M Dataset Implementation
# -----------------------------------------------------------------------------
class LmsysChatDataset(BenchmarkDataset):
    '''
    Implements Lmsys-chat-1M dataset.
    '''
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()
        
    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
    
        # Load parquet file using pandas
        self.data = pd.read_parquet(self.dataset_path)
    
        # Filter entries that have messages column and at least one message
        if "conversation" not in self.data.columns:
            raise ValueError("Dataset must have a 'conversation' column.")
        
        # Filter entries with at least one message
        self.data = self.data[self.data["conversation"].notna()]
        
        # Convert messages to list if it's stored as JSON string or other types
        def parse_messages(x):
            # Handle None/NaN
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            
            # Handle string (JSON)
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return None
            
            # Handle numpy array (common in parquet files)
            if hasattr(x, '__iter__') and not isinstance(x, (str, dict)):
                try:
                    # Convert numpy array, PyArrow ListArray, or other iterables to list
                    if hasattr(x, 'tolist'):
                        result = x.tolist()
                    elif hasattr(x, 'to_pylist'):
                        result = x.to_pylist()
                    else:
                        result = list(x)
                        print("this is list")

                    # Validate that result is a list of dicts
                    if isinstance(result, list) and len(result) > 0:
                        # Check if first element is a dict (valid message format)
                        if isinstance(result[0], dict):
                            return result
                        # If it's a list of lists, flatten it
                        if isinstance(result[0], list):
                            return [item for sublist in result for item in sublist if isinstance(item, dict)]
                    return result
                except (TypeError, AttributeError, ValueError):
                    return None
            
            # Handle list-like types (list, tuple, etc.)
            if isinstance(x, (list, tuple)):
                return list(x)
            
            # Handle dict (single message wrapped in dict)
            if isinstance(x, dict):
                return [x] if "role" in x and "content" in x else None
            
            # Return as-is if already a list
            return x
        
        self.data["conversation"] = self.data["conversation"].apply(parse_messages)   
        
        # Filter: keep only entries with valid list of messages
        def is_valid_messages(x):
            if x is None:
                return False
            if not isinstance(x, list):
                return False
            if len(x) == 0:
                return False
            # Check if all items are dicts with required keys
            return all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in x)
        
        self.data = self.data[self.data["conversation"].apply(is_valid_messages)]
        
        # Convert to list of dicts for easier iteration
        self.data = self.data.to_dict("records")
        
        # Shuffle the list after conversion
        random.seed(self.random_seed)
        random.shuffle(self.data)
        
    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            total_req_num = len(requests) 
            orig_requests = requests[:]
            copy_time = num_requests // total_req_num
            remain = num_requests % total_req_num
            for i in range(copy_time):
                if (i == copy_time - 1) and (remain != 0):
                    requests.extend(orig_requests[:remain])
                else:
                    requests.extend(orig_requests)
            
            assert len(requests) == num_requests, f"should assert # request to {num_requests}, but only {len(requests)} was ready"
            
            
    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break

            messages = entry["conversation"]
            if not isinstance(messages, list) or len(messages) == 0:
                continue
            
            # Validate message format
            if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
                continue
            
            # Find the last assistant message index
            last_assistant_idx = -1
            for i in range(len(messages) - 1, -1, -1):
               if messages[i].get("role") == "assistant":
                   last_assistant_idx = i
                   break
            
            # If no assistant message found, skip this entry
            if last_assistant_idx == -1:
                continue
            
            # Get all messages up to (but not including) the last assistant message
            prompt_messages = messages[:last_assistant_idx]
            
            # Skip if prompt_messages is empty (last assistant is the first message)
            if len(prompt_messages) == 0:
                continue
            
            # Convert to chat format using tokenizer's apply_chat_template
            # If there's no system message, add a default one
            if prompt_messages[0].get("role") != "system":
                conversation = [{"role": "system", "content": "You are a helpful assistant."}]
                conversation.extend(prompt_messages)
            else:
                conversation = prompt_messages
            
            # Apply chat template to create the prompt
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            prompt_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_ids)
            
            # Use provided output_len or a default value
            new_output_len = output_len if output_len is not None else 512
            
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                max_total_len=4096,
                max_prompt_len=2048,                
                skip_min_output_len_check=output_len is not None
            ):
                continue
                        
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                )
            )
        self.maybe_oversample_requests(samples, num_requests)
        return samples
# -----------------------------------------------------------------------------
# SpecBench Dataset Implementation
        return samples