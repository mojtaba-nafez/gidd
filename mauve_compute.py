'''
python mauve_compute.py samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=8 metrics_path=samples.json
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
from gidd.data import get_dataloaders
import mauve
import os

def empirical_entropy(items):
    counts = Counter(items)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    return -sum((c / total) * math.log(c / total) for c in counts.values())

def empirical_entropy_per_sample(items):
    entropies = []
    for row in items:
        row = torch.tensor(row)
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

@hydra.main(config_path="gidd/configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)
    # Load generated token IDs
    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    # Fix extra singleton dimension from self-correct script
    if z_ts.ndim == 3 and z_ts.shape[1] == 1:
        z_ts = z_ts.squeeze(1)
    print("z_ts.shape:", z_ts.shape)
    # Token IDs -> text
    generated_texts = tokenizer.batch_decode(
        z_ts.cpu().tolist(),
        skip_special_tokens=True,
    )
    # Load human references
    _, test_dl = get_dataloaders(args, tokenizer)
    print(f"Loaded test dataloader with {len(test_dl.dataset)} samples")
    human_ids = torch.tensor(
        test_dl.dataset[:len(generated_texts)]["input_ids"]
    )
    print("human_references.shape:", human_ids.shape)
    # Token IDs -> text
    human_texts = tokenizer.batch_decode(
        human_ids.tolist(),
        skip_special_tokens=True,
    )

    # MAUVE expects list[str], not token IDs
    results = mauve.compute_mauve(
        p_text=human_texts,
        q_text=generated_texts,
        device_id=0,
        max_text_length=512,
        verbose=False,
    )

    mauve_score = float(results.mauve)
    print("Mauve results:", mauve_score)

    # Save next to args.samples_path
    output_path = os.path.splitext(samples_path)[0] + "_mauve.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "mauve": mauve_score,
                "samples_path": samples_path,
            },
            f,
            indent=4,
        )

    print(f"MAUVE result saved to: {output_path}")
    # generated_token_ids = []
    # samples_generated_token_ids = []
    # for text in texts:
    #     token_ids = model_tokenizer.encode(text, add_special_tokens=False)
    #     samples_generated_token_ids.append(token_ids)
    # print("torch.tensor(samples_generated_token_ids).shape: ", torch.tensor(samples_generated_token_ids).shape)
    '''
    unigram_entropy = empirical_entropy(generated_token_ids)
    distinct_1 = distinct_n(generated_token_ids)
    unigram_entropy_per_sample = empirical_entropy_per_sample(samples_generated_token_ids)

    total_acc = 0
    total_nll = 0
    total_tokens = 0
    all_nlls = []
    per_sample = []
    with torch.no_grad():
        for i in tqdm.trange(0, len(texts), args.batch_size, desc="Inference", dynamic_ncols=True):
            xs = texts[i:i + args.batch_size]

            batch = tokenizer(xs, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
            attn_mask = batch["attention_mask"]
        
            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]

            labels = batch["input_ids"][:, 1:]
            loss_mask = attn_mask[:, :-1]

            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
            all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
            total_nll += (nll * loss_mask).sum().item()

            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()

            total_tokens += loss_mask.sum().item()

            sample_nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
            sample_ppl = torch.exp(sample_nll)

            for text, ppl_i, entropy_i in zip(xs, sample_ppl.cpu().tolist(), unigram_entropy_per_sample[i:i + args.batch_size]):
                per_sample.append({
                    "ppl": ppl_i,
                    "unigram_entropy": entropy_i,
                    "text": text
                })

    nll = total_nll / total_tokens
    ppl = np.exp(total_nll / total_tokens)
    acc = total_acc / total_tokens

    metrics = {
        "file": Path(args.samples_path).stem,
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
    '''

if __name__ == "__main__":
    main()
