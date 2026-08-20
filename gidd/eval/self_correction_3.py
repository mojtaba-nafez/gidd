import pandas as pd
import hydra
import tqdm
import torch

from gidd.checkpoints import load_checkpoint
from gidd.utils import parse_dtype, sample_categorical


def correction_step_original(model, tokenizer, z_t, t, temp, tokens_per_step, latent_noise=False, activate_nvib_noise=False, use_trained_scaling_factor=False, rm_self_attention=False):
    """Original GIDD self-correction. Kept only for comparison / ablation."""
    logits = model(z_t, t, latent_noise=latent_noise, activate_nvib_noise=activate_nvib_noise, use_trained_scaling_factor=use_trained_scaling_factor, rm_self_attention=rm_self_attention)
    logits = logits.clone()
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)
    z_tm1 = sample_categorical(p_t)

    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)
    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
    return z_tm1, acc

def correction_step(model, tokenizer, z_t, t, temp, tokens_per_step=1, step=0, rand_tokens_per_step=1, latent_noise=False, activate_nvib_noise=False, use_trained_scaling_factor=False, rm_self_attention=False):
    """Hybrid self-correction step:
    1. Logit-based Correction: Picks `tokens_per_step` positions where the model disagrees 
       most with `z_t` and resamples tokens using temperature-scaled model logits.
    2. Random Perturbation: Picks `rand_tokens_per_step` additional valid positions completely 
       at random and replaces them with uniform tokens from the vocabulary.
    """
    if temp <= 0:
        raise ValueError(f"Temperature must be > 0, got {temp}")

    # --- Step 1: Model Forward Pass & Logit Evaluation ---
    logits = model(z_t, t, latent_noise=latent_noise, activate_nvib_noise=activate_nvib_noise, use_trained_scaling_factor=use_trained_scaling_factor, rm_self_attention=rm_self_attention)
    logits = logits.clone()
    logits[..., tokenizer.mask_token_id] = -torch.inf

    pred_tokens = logits.argmax(dim=-1)
    self_acc = (z_t == pred_tokens).float().mean().item()

    best_logits = logits.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)
    current_logits = logits.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    margin = best_logits - current_logits

    # Base valid mask for non-pad, non-mask tokens
    valid_mask = torch.ones_like(z_t, dtype=torch.bool)
    if tokenizer.pad_token_id is not None:
        valid_mask = valid_mask & (z_t != tokenizer.pad_token_id)
    valid_mask = valid_mask & (z_t != tokenizer.mask_token_id)

    correctable = (pred_tokens != z_t) & valid_mask
    num_correctable = correctable.sum().item()

    z_next = z_t.clone()
    total_changed = 0
    mean_margin = 0.0

    # --- Step 2: Logit-Based Correction ---
    logit_selected_ids = None
    if num_correctable > 0 and tokens_per_step > 0:
        score = margin.masked_fill(~correctable, -torch.inf)
        k_logit = min(int(tokens_per_step), z_t.shape[-1], int(num_correctable))
        top_scores, logit_selected_ids = torch.topk(score, k=k_logit, dim=-1)
        valid_logit = torch.isfinite(top_scores)

        vocab_size = logits.shape[-1]
        selected_logits = logits.gather(1, logit_selected_ids.unsqueeze(-1).expand(-1, -1, vocab_size)).clone()
        current_selected = z_t.gather(-1, logit_selected_ids)

        # Block re-sampling the current token
        selected_logits.scatter_(-1, current_selected.unsqueeze(-1), -torch.inf)

        selected_probs = torch.softmax(selected_logits.float() / temp, dim=-1)
        sampled_tokens = sample_categorical(selected_probs)
        sampled_tokens = torch.where(valid_logit, sampled_tokens, current_selected)

        z_next = z_next.scatter(-1, logit_selected_ids, sampled_tokens)
        changed_logit = valid_logit & (sampled_tokens != current_selected)
        total_changed += changed_logit.sum().item()

        valid_scores = top_scores[valid_logit]
        mean_margin = valid_scores.mean().item() if valid_scores.numel() > 0 else 0.0

    # --- Step 3: Random Replacement ---
    if rand_tokens_per_step > 0:
        rand_valid_mask = valid_mask.clone()
        # Exclude positions already updated by the logit-based correction step
        if logit_selected_ids is not None:
            rand_valid_mask.scatter_(-1, logit_selected_ids, False)

        num_rand_valid = rand_valid_mask.sum().item()
        if num_rand_valid > 0:
            k_rand = min(int(rand_tokens_per_step), z_t.shape[-1], int(num_rand_valid))

            # Assign random keys to valid positions and pick top-k
            rand_keys = torch.rand_like(z_t, dtype=torch.float32)
            rand_keys = rand_keys.masked_fill(~rand_valid_mask, -1.0)
            _, rand_ids = torch.topk(rand_keys, k=k_rand, dim=-1)

            vocab_size = logits.shape[-1]
            random_sampled_tokens = torch.randint(0, vocab_size, size=rand_ids.shape, device=z_t.device, dtype=z_t.dtype)

            current_rand_selected = z_next.gather(-1, rand_ids)
            z_next = z_next.scatter(-1, rand_ids, random_sampled_tokens)

            changed_rand = (random_sampled_tokens != current_rand_selected)
            total_changed += changed_rand.sum().item()

    has_correctable = (num_correctable > 0) or (rand_tokens_per_step > 0 and valid_mask.sum().item() > 0)

    if not has_correctable:
        return {
            "z_next": z_t.clone(), "self_acc": self_acc, "num_changed": 0,
            "num_correctable": 0, "has_correctable": False, "mean_margin": 0.0,
            "status": "fixed_point",
        }

    return {
        "z_next": z_next, 
        "self_acc": self_acc, 
        "num_changed": total_changed,
        "num_correctable": num_correctable, 
        "has_correctable": True, 
        "mean_margin": mean_margin,
        "status": "changed" if total_changed > 0 else "no_op",
    }


def escape_fixed_point(logits, tokenizer, z_t, num_escape_tokens=1, escape_temp=0.0):
    """Escape a true GIDD fixed point (every editable position already equals the model's
    argmax, so normal self-correction has no candidate). Finds the LEAST STABLE positions —
    smallest (current_logit - best_alternative_logit) — and force-replaces a few so ordinary
    correction can resume. escape_temp<=0 uses the deterministic second-best token; >0 samples
    from the alternatives (current token excluded)."""
    logits = logits.clone()
    logits[..., tokenizer.mask_token_id] = -torch.inf
    batch_size, seq_len, vocab_size = logits.shape

    current_logits = logits.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)

    alternative_logits = logits.clone()
    alternative_logits.scatter_(-1, z_t.unsqueeze(-1), -torch.inf)
    best_alt_logits, best_alt_tokens = alternative_logits.max(dim=-1)

    stability_margin = current_logits - best_alt_logits

    valid_position = z_t != tokenizer.mask_token_id
    if tokenizer.pad_token_id is not None:
        valid_position = valid_position & (z_t != tokenizer.pad_token_id)
    stability_margin = stability_margin.masked_fill(~valid_position, torch.inf)
    num_valid = valid_position.sum().item()

    if num_valid == 0:
        return {"z_next": z_t.clone(), "escape_positions": None, "escape_margin": float("inf"), "num_changed": 0}

    k = min(int(num_escape_tokens), seq_len, int(num_valid))
    escape_margins, ids = torch.topk(stability_margin, k=k, dim=-1, largest=False)
    valid_escape = torch.isfinite(escape_margins)
    current_selected = z_t.gather(-1, ids)

    if escape_temp <= 0:
        replacement = best_alt_tokens.gather(-1, ids)
    else:
        selected_alt_logits = alternative_logits.gather(1, ids.unsqueeze(-1).expand(-1, -1, vocab_size))
        selected_probs = torch.softmax(selected_alt_logits.float() / escape_temp, dim=-1)
        replacement = sample_categorical(selected_probs)

    replacement = torch.where(valid_escape, replacement, current_selected)
    z_next = z_t.scatter(-1, ids, replacement)

    changed = (replacement != current_selected) & valid_escape
    num_changed = changed.sum().item()
    mean_escape_margin = escape_margins[valid_escape].mean().item() if valid_escape.any() else float("inf")

    return {"z_next": z_next, "escape_positions": ids, "escape_margin": mean_escape_margin, "num_changed": num_changed}


def evaluate_self_accuracy(model, tokenizer, z_t, t, latent_noise=False, activate_nvib_noise=False, use_trained_scaling_factor=False, rm_self_attention=False):
    """Evaluate self-accuracy using exactly the same model options as the correction step."""
    logits = model(z_t, t, latent_noise=latent_noise, activate_nvib_noise=activate_nvib_noise, use_trained_scaling_factor=use_trained_scaling_factor, rm_self_attention=rm_self_attention)
    logits = logits.clone()
    logits[..., tokenizer.mask_token_id] = -torch.inf
    acc = (z_t == logits.argmax(-1)).float().mean().item()
    return logits, acc


@hydra.main(config_path="../configs", config_name="self_correction", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.path)
    print("args.latent_noise", args.latent_noise)
    print("args.activate_nvib_noise", args.activate_nvib_noise)
    print("args.use_trained_scaling_factor", args.use_trained_scaling_factor)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)
    model = torch.compile(model)

    # Bundled once: identical for every call below, never mutated
    model_kwargs = dict(latent_noise=args.latent_noise, activate_nvib_noise=args.activate_nvib_noise,
                         use_trained_scaling_factor=args.use_trained_scaling_factor, rm_self_attention=args.rm_self_attention)

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    # Debug: uncomment to test only sample index 32
    # z_ts = z_ts[32:33, :]

    # Fixed-point escape params (Hydra-overridable, e.g. ++max_fixed_point_escapes=3)
    max_fixed_point_escapes = int(getattr(args, "max_fixed_point_escapes", 3))
    escape_num_tokens = int(getattr(args, "escape_num_tokens", 1))
    escape_temp = float(getattr(args, "escape_temp", 0.0))

    print()
    print("=== FIXED-POINT ESCAPE SETTINGS ===")
    print("max_fixed_point_escapes:", max_fixed_point_escapes)
    print("escape_num_tokens:", escape_num_tokens)
    print("escape_temp:", escape_temp)
    print("===================================")
    print()

    metrics = []
    samples = []

    for sample_idx, z_t in enumerate(tqdm.tqdm(z_ts, desc="Correction", dynamic_ncols=True, smoothing=0.0)):
        z_t = z_t.unsqueeze(0).to(device)
        z_t_init = z_t.clone()
        t = torch.full((z_t.shape[0],), device=device, fill_value=args.t0)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            init_logits, init_acc = evaluate_self_accuracy(model=model, tokenizer=tokenizer, z_t=z_t, t=t, **model_kwargs)

        max_acc = init_acc
        curr_patience = 0
        converged = 0
        early_stopped = 0
        num_fixed_point_escapes = 0
        total_escape_changes = 0
        total_edit_steps = 0
        total_token_edits = 0
        last_acc = init_acc
        escape_margins = []

        for i in range(args.num_denoising_steps):
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                out = correction_step(model=model, tokenizer=tokenizer, z_t=z_t, t=t, temp=args.temp,
                                       tokens_per_step=args.tokens_per_step, rand_tokens_per_step=args.rand_tokens_per_step, step=i, **model_kwargs)

            z_t_next = out["z_next"]
            acc = out["self_acc"]
            last_acc = acc

            if not out["has_correctable"]:
                print()
                print(f"[sample {sample_idx}] FIXED POINT at step {i} (self_acc={acc:.6f})")

                if num_fixed_point_escapes < max_fixed_point_escapes:
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                        escape_logits = model(z_t, t, **model_kwargs)

                    escape_out = escape_fixed_point(logits=escape_logits, tokenizer=tokenizer, z_t=z_t,
                                                      num_escape_tokens=escape_num_tokens, escape_temp=escape_temp)

                    if escape_out["num_changed"] == 0:
                        print(f"[sample {sample_idx}] Fixed-point escape failed.")
                        converged = 1
                        break

                    positions = escape_out["escape_positions"]
                    if positions is not None:
                        for j in range(positions.shape[-1]):
                            pos = positions[0, j].item()
                            old_id = z_t[0, pos].item()
                            new_id = escape_out["z_next"][0, pos].item()
                            try:
                                old_text = tokenizer.decode([old_id])
                                new_text = tokenizer.decode([new_id])
                            except Exception:
                                old_text = str(old_id)
                                new_text = str(new_id)
                            print(f"[sample {sample_idx}] ESCAPE #{num_fixed_point_escapes + 1}: position={pos}, "
                                  f"old={old_text!r}, new={new_text!r}, margin={escape_out['escape_margin']:.6f}")

                    z_t = escape_out["z_next"]
                    num_fixed_point_escapes += 1
                    total_escape_changes += escape_out["num_changed"]
                    escape_margins.append(escape_out["escape_margin"])

                    # CRITICAL: the prior fixed point may have had self_acc=1.0 — reset patience so
                    # post-escape states aren't blocked from ever counting as an "improvement"
                    max_acc = -float("inf")
                    curr_patience = 0
                    continue

                print(f"[sample {sample_idx}] Fixed-point escape budget exhausted.")
                converged = 1
                break

            # Stochastic no-op: rare now that the current token is excluded from resampling
            if out["num_changed"] == 0:
                print(f"[sample {sample_idx}] NO-OP at step {i}; retrying.")
                continue

            z_t = z_t_next
            total_edit_steps += 1
            total_token_edits += out["num_changed"]

            # Patience logic: acc is the pre-update state; the updated state is evaluated next loop
            if acc > max_acc:
                max_acc = acc
                curr_patience = 0
            else:
                curr_patience += 1
                if curr_patience > args.max_patience:
                    early_stopped = 1
                    print(f"[sample {sample_idx}] EARLY STOP at step {i}")
                    break

        # Evaluate final state — last_acc may be pre-final-edit
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            final_logits, final_acc = evaluate_self_accuracy(model=model, tokenizer=tokenizer, z_t=z_t, t=t, **model_kwargs)

        num_changes = (z_t_init != z_t).sum().item()
        samples.append(z_t)

        metrics.append({
            "sample_idx": sample_idx,
            "init_acc": init_acc,
            "final_acc": final_acc,
            "improvement": final_acc - init_acc,
            "num_changes": num_changes,  # net diff vs. init
            "total_token_edits": total_token_edits,  # total edits across trajectory
            "total_edit_steps": total_edit_steps,
            "fixed_point_escapes": num_fixed_point_escapes,
            "escape_token_changes": total_escape_changes,
            "mean_escape_margin": sum(escape_margins) / len(escape_margins) if escape_margins else 0.0,
            "converged": converged,
            "early_stopped": early_stopped,
        })

    samples = torch.cat(samples, dim=0).cpu()
    corrected_samples_path = hydra.utils.to_absolute_path(args.corrected_samples_path)
    print()
    print("Saving corrected samples:", corrected_samples_path)
    torch.save(samples, corrected_samples_path)

    df = pd.DataFrame(metrics)
    metrics_path = hydra.utils.to_absolute_path(args.metrics_path)
    df.to_csv(metrics_path, index=False)

    print()
    print(f"Results for {args.path} (temp={args.temp}, max_patience={args.max_patience})")
    print()
    print(df.describe().to_markdown())
    print()
    print("Mean init acc:", df["init_acc"].mean())
    print("Mean final acc:", df["final_acc"].mean())
    print("Mean improvement:", df["improvement"].mean())
    print("Samples requiring fixed-point escape:", (df["fixed_point_escapes"] > 0).sum())
    print("Total fixed-point escapes:", df["fixed_point_escapes"].sum())
    print("Total escape token changes:", df["escape_token_changes"].sum())


if __name__ == "__main__":
    main()