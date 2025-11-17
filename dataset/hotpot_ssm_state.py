import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from hotpot import HotpotQAIterator

# Add mamba to path
sys.path.insert(0, '../mamba')
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba.mamba_ssm.utils.generation import InferenceParams
from transformers import AutoTokenizer
from tqdm import tqdm


def extract_cache_single(model, tokenizer, prompt, device="cuda", return_tensor=True):
    """
    Extract cache from a single prompt using pre-loaded model and tokenizer.

    Args:
        model: Pre-loaded MambaLMHeadModel
        tokenizer: Pre-loaded tokenizer
        prompt: Input text prompt
        device: Device to use
        return_tensor: If True, return stacked SSM states; else return InferenceParams

    Returns:
        If return_tensor=True: torch.Tensor of shape [num_layers, d_inner, d_state]
        If return_tensor=False: InferenceParams object
    """
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    inference_params = InferenceParams(
        max_seqlen=prompt_len + 8,
        max_batch_size=batch_size
    )

    with torch.no_grad():
        logits = model(input_ids, inference_params=inference_params).logits

    if return_tensor:
        kv_dict = inference_params.key_value_memory_dict
        num_layers = len(kv_dict)

        ssm_states = [kv_dict[layer_idx][1].squeeze(0) for layer_idx in sorted(kv_dict.keys())]
        stacked_states = torch.stack(ssm_states, dim=0)
        return stacked_states
    else:
        return inference_params


class HotpotDoc1CacheIterator:
    """
    HotpotQA Doc1 Cache Iterator for Mamba
    """

    def __init__(self, json_path, gpu_id=0, num_samples=10, random_seed=42, external_model=None, external_tokenizer=None):
        self._index = 0

        # Tackle sampling and random seed
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.hotpot_iterator = HotpotQAIterator(json_path).random_choose(num_samples, seed=random_seed)

        # Record device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Use external model/tokenizer if provided, otherwise load own
        if external_model is not None and external_tokenizer is not None:
            self.model = external_model
            self.tokenizer = external_tokenizer
            self.owns_model = False
            print(f"Using external model on {self.device}")
        else:
            # Load model and tokenizer
            self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.owns_model = True
            print(f"✓ Model loaded on {self.device} (GPU {gpu_id})")

    def create_doc1_prompt(self, item):
        doc1 = item.context[0]
        return f"Document 1: {doc1.title}\n{doc1.get_full_text()}\n\n"

    def extract_doc1_cache(self, item):
        prompt = self.create_doc1_prompt(item)
        cache_tensor = extract_cache_single(self.model, self.tokenizer, prompt, device=self.device, return_tensor=True)
        return cache_tensor

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.hotpot_iterator):
            raise StopIteration

        item = self.hotpot_iterator[self._index]
        self._index += 1

        return self.extract_doc1_cache(item)

    def __len__(self):
        return len(self.hotpot_iterator)
    
class HotpotDoc1PlusCacheIterator:
    def __init__(self, json_path, plus_content, gpu_id=0, num_samples=10, random_seed=42, external_model=None, external_tokenizer=None):
        self._index = 0

        self.plus_content = plus_content

        # Tackle sampling and random seed
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.hotpot_iterator = HotpotQAIterator(json_path).random_choose(num_samples, seed=random_seed)

        # Record device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Use external model/tokenizer if provided, otherwise load own
        if external_model is not None and external_tokenizer is not None:
            self.model = external_model
            self.tokenizer = external_tokenizer
            self.owns_model = False
        else:
            # Load model and tokenizer
            self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.owns_model = True

    def create_doc1_prompt(self, item):
        doc1 = item.context[0]
        return f"Document 1: {doc1.title}\n{doc1.get_full_text()}\n\n"

    def extract_doc1_plus_cache(self, item):
        prompt = self.create_doc1_prompt(item) + self.plus_content
        cache_tensor = extract_cache_single(self.model, self.tokenizer, prompt, device=self.device, return_tensor=True)
        return cache_tensor

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.hotpot_iterator):
            raise StopIteration

        item = self.hotpot_iterator[self._index]
        self._index += 1

        return self.extract_doc1_plus_cache(item)

    def __len__(self):
        return len(self.hotpot_iterator)

class HotpotDoc2CacheIterator:
    def __init__(self, json_path, gpu_id=0, num_samples=10, random_seed=42):
        self._index = 0

        # Tackle sampling and random seed
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.hotpot_iterator = HotpotQAIterator(json_path).random_choose(num_samples, seed=random_seed)

        # Record device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b", device=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    def create_doc2_prompt(self, item):
        doc2 = item.context[1]
        return f"Document 2: {doc2.title}\n{doc2.get_full_text()}\n\n"

    def extract_doc2_cache(self, item):
        prompt = self.create_doc2_prompt(item)
        cache_tensor = extract_cache_single(self.model, self.tokenizer, prompt, device=self.device, return_tensor=True)
        return cache_tensor

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.hotpot_iterator):
            raise StopIteration

        item = self.hotpot_iterator[self._index]
        self._index += 1

        return self.extract_doc1_cache(item)

    def __len__(self):
        return len(self.hotpot_iterator)


if __name__ == "__main__":
    json_path = "dataset/HotpotQA/hotpot_train_v1.1.json"
    iterator = HotpotDoc1CacheIterator(json_path, gpu_id=0, num_samples=1000, random_seed=42)

    for idx, cache_tensor in enumerate(tqdm(iterator, total=len(iterator))):
        continue
        print(f"Sample {idx}: Cache Tensor Shape: {cache_tensor.shape}")
    
    
