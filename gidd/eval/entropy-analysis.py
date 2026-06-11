import json
from pathlib import Path

import hydra
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import GPT2TokenizerFast
from collections import Counter
import math

def empirical_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    return -sum((c / total) * math.log(c / total) for c in counts.values())

def empirical_entropy_per_sample(items, low_threshold=0, up_threshold=1024):
    entropies = []
    for row in items:
        row = torch.tensor(row)
        if not (len(row) < up_threshold and len(row) >= low_threshold):
            continue
        counts = torch.unique(
            row,
            return_counts=True,
            sorted=True
        )[1]
        probs = counts.float() / counts.sum()
        entropy = torch.special.entr(probs).sum().item()
        entropies.append(entropy)
    return entropies


def distinct_n(items):
    if len(items) == 0:
        return 0.0
    return len(set(items)) / len(items)

@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    print(f"Loding model {args.pretrained_model}")

    # model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if args.torch_compile:
        # model = torch.compile(model)

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    # fix for bug in self-correct script:
    if z_ts.shape[1] == 1:
        z_ts = z_ts.squeeze(1)
    texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=False)
    # Diversity metrics over generated samples.
    # Use the same tokenizer that produced the samples, so the entropy is comparable
    # across runs that use the same generation tokenizer.
    generated_token_ids = []
    samples_generated_token_ids = []
    ii = 0
    kk = 0
    print("=====================")
    for text in texts:
        token_ids = model_tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids)<10:
            print(f"index {ii}", len(token_ids))
            print(text)
            print("=====================")
            kk += 1

        generated_token_ids.extend(token_ids)
        samples_generated_token_ids.append(token_ids)
        ii +=1

    print("number of shit generated sentences:", kk)

    # unigram_entropy = empirical_entropy(generated_token_ids)
    # distinct_1 = distinct_n(generated_token_ids)
    unigram_entropy_per_sample = empirical_entropy_per_sample(samples_generated_token_ids, low_threshold=args.entropy_sample_len_low_threshold, up_threshold=args.entropy_sample_len_up_threshold)

    print("unigram_entropy_per_sample", sum(unigram_entropy_per_sample) / len(unigram_entropy_per_sample))
    print("len(unigram_entropy_per_sample)",  f"{len(unigram_entropy_per_sample)} out of {len(texts)} sample")
    
    print(f"range of sample lens(context_len=512):  {args.entropy_sample_len_low_threshold} < sample len < {args.entropy_sample_len_up_threshold}")
    
    return 

if __name__ == "__main__":
    main()
