'''
python gidd/eval/get_ppl_training_data.py model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 \
    metrics_path=training_dataset.json +training.train_batch_size=1 +training.eval_batch_size=1 +data.test_size=100000 \
    +data.dataset_name=Skylion007/openwebtext +data.dataset_subset=null +data.trust_remote_code=true +data.tokenizer_name=gpt2 \
    +model.max_seq_len=512 +data.sequence_packing=false +data.max_add_padding=0 +data.cache_dir=./cache +data.num_workers=4
'''
import json
from pathlib import Path

import hydra
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import math
from contextlib import contextmanager
import torch.distributed as dist
from gidd.data import get_dataloaders



def empirical_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    return -sum((c / total) * math.log(c / total) for c in counts.values())

def empirical_entropy_per_sample(items):
    entropies = []

    for row in items:
        # row = row[row != 50256]
        # if len(row) <= 450:
            # continue
        counts = torch.bincount(row, minlength=50257)
        counts = counts[counts > 0]

        probs = counts.float() / counts.sum()
        entropies.append(torch.special.entr(probs).sum().item())

    return entropies

# def empirical_entropy_per_sample(items):
#     entropies = []
#     for row in items:
#         row = row[row != 50256]
#         # if len(row) <= 450:
#         #     # print("low that 450")
#         #     continue
#         counts = torch.unique(
#             row,
#             return_counts=True,
#             sorted=False
#         )[1]
#         probs = counts.float() / counts.sum()
#         entropy = torch.special.entr(probs).sum().item()
#         entropies.append(entropy)
#     return entropies


def distinct_n(items):
    if len(items) == 0:
        return 0.0
    return len(set(items)) / len(items)

@contextmanager
def main_process_first():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    else:
        yield


@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    data_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        device_map="auto"
    )
    gamma_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if gamma_tokenizer.pad_token_id is None:
        gamma_tokenizer.pad_token = gamma_tokenizer.eos_token
    if args.torch_compile:
        model = torch.compile(model)

    with main_process_first():
        train_dl, test_dl = get_dataloaders(args, data_tokenizer)
    
    unigram_entropy_per_sample = []
    batch_num = 0
    token_counter = Counter()

    for batch in tqdm.tqdm(train_dl, desc="Entropy"):
        input_ids = batch["input_ids"] #  input_ids.shape: torch.Size([1, 512])
        for seq in input_ids:
            seq = seq.tolist()
            token_counter.update(seq)
        unigram_entropy_per_sample.extend(empirical_entropy_per_sample(input_ids))
        batch_num += 1
        if batch_num > 100000:
            break

    token_counter.pop(50256, None)
    total_tokens = sum(token_counter.values())
    unigram_entropy = -sum(
        (c / total_tokens) * math.log(c / total_tokens)
        for c in token_counter.values()
    ) if total_tokens > 0 else 0.0
    distinct_1 = len(token_counter) / max(total_tokens, 1)
    avg_entropy_per_sample = sum(unigram_entropy_per_sample) / len(unigram_entropy_per_sample) if len(unigram_entropy_per_sample) > 0 else 0.0
    print(f"Average entropy per sample: {avg_entropy_per_sample} over {len(unigram_entropy_per_sample)} samples out of {batch_num*args.batch_size} total samples")
    
    return
    
    total_acc = 0
    total_nll = 0
    total_tokens = 0
    all_nlls = []
    per_sample = []
    batch_num = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(train_dl, desc="Inference PPL"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            texts = data_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            batch = gamma_tokenizer(texts, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
            attn_mask = batch["attention_mask"]

            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]

            labels = batch["input_ids"][:, 1:].to(device)
            loss_mask = attn_mask[:, :-1]

            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
            all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
            total_nll += (nll * loss_mask).sum().item()

            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()

            total_tokens += loss_mask.sum().item()

            sample_nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
            sample_ppl = torch.exp(sample_nll)

            for text, ppl_i, entropy_i in zip(texts, sample_ppl.cpu().tolist(), unigram_entropy_per_sample[batch_num*args.batch_size:batch_num*args.batch_size + args.batch_size]):
                
                per_sample.append({
                    "ppl": ppl_i,
                    "unigram_entropy": entropy_i,
                    "text": text
                })
            batch_num += 1
            if batch_num > 2000:
                break

    nll = total_nll / total_tokens
    ppl = np.exp(total_nll / total_tokens)
    acc = total_acc / total_tokens

    metrics = {
        "pretrained_model": args.pretrained_model,
        "median_nll": np.median(all_nlls),
        "avg_nll": nll,
        "ppl": ppl,
        "acc": acc,
        "tokens": total_tokens,

        # Diversity metrics
        "unigram_entropy": unigram_entropy,
        "distinct_1": distinct_1,
        "unigram_entropy_per_sample": sum(unigram_entropy_per_sample) / len(unigram_entropy_per_sample),

        "per_sample": per_sample
    }

    json.dumps(metrics, indent=4)
    print("=== RESULTS ===")
    print("\n".join(map(str, [
        f"ppl={metrics['ppl']}",
        f"acc={metrics['acc']}",
        f"unigram_entropy={metrics['unigram_entropy']}",
        f"unigram_entropy_per_sample={metrics['unigram_entropy_per_sample']}",
    ])))
    print("===============")

    with open(hydra.utils.to_absolute_path(args.metrics_path), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()