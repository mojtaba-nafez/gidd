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

@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    print(f"Loding model {args.pretrained_model}")

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.torch_compile:
        model = torch.compile(model)

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    # fix for bug in self-correct script:
    if z_ts.shape[1] == 1:
        z_ts = z_ts.squeeze(1)
    texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)
    # Diversity metrics over generated samples.
    # Use the same tokenizer that produced the samples, so the entropy is comparable
    # across runs that use the same generation tokenizer.
    generated_token_ids = []
    samples_generated_token_ids = []
    for text in texts:
        token_ids = model_tokenizer.encode(text, add_special_tokens=False)
        generated_token_ids.extend(token_ids)
        samples_generated_token_ids.append(token_ids)

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
    # print("=== RESULTS ===")
    # print(",".join(map(str, [
    #     metrics["file"],
    #     metrics["pretrained_model"],
    #     metrics["median_nll"],
    #     metrics["avg_nll"],
    #     metrics["ppl"],
    #     metrics["acc"],
    #     metrics["tokens"],
    #     metrics["unigram_entropy"],
    #     metrics["distinct_1"],
    #     metrics["unigram_entropy_per_sample"],
    # ])))
    # print("===============")

    with open(hydra.utils.to_absolute_path(args.metrics_path), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
