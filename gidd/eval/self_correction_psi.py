"""Post-hoc GIDD self-correction with Psi-inspired forward re-noising.

This script is intentionally a DROP-IN replacement for gidd/eval/self_correction.py:
it loads already generated token tensors from ``samples_path`` and writes a corrected
``.pt`` file.

Important terminology
---------------------
The ICLR-2026 Psi sampler is a predictor-corrector *generation* sampler.  Its exact
transition mixes the ancestral posterior with a branch that goes to the clean
posterior and is then forward-noised again.  Once a fully generated x_0 sample has
already been saved, that exact transition cannot be retroactively applied without
putting the sample back onto a diffusion trajectory.

Here we adapt the same error-correction idea to GIDD's original post-hoc fixed-point
self-correction:

  * ordinary GIDD self-correction edits tokens the model judges to be wrong;
  * when the iteration reaches a fixed point / stagnates, instead of forcing a
    deterministic second-best token, we sample a SMALL forward-noising move from
    GIDD's own trained hybrid kernel q_t(z|x) = alpha_t x + beta_t pi_t;
  * ordinary self-correction then resumes and denoises/repairs those perturbations.

For GIDD p_uniform > 0 this naturally injects mostly random visible-token
perturbations plus the schedule-appropriate mask probability.  This is much closer
to Psi's "re-introduce forward noise so errors can become revisable" mechanism than
an ad-hoc random vocabulary replacement, while preserving the user's existing
``generated .pt -> self correction -> PPL`` workflow.

For a *mathematically exact* Psi experiment, integrate Psi into generate_samples.py
and regenerate the samples.  Treat this file as "Psi-inspired post-hoc refinement",
not as a claim that the original Psi posterior is being sampled after generation.
"""

from __future__ import annotations

import math
import pandas as pd
import hydra
import tqdm
import torch

from gidd.checkpoints import load_checkpoint
from gidd.utils import parse_dtype, sample_categorical


def _call_model(model, z_t, t, *, latent_noise=False,
                activate_nvib_noise=False,
                use_trained_scaling_factor=False,
                rm_self_attention=False):
    return model(
        z_t,
        t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
        rm_self_attention=rm_self_attention,
    )


def correction_step_original(
    model,
    tokenizer,
    z_t,
    t,
    temp,
    tokens_per_step,
    latent_noise=False,
    activate_nvib_noise=False,
    use_trained_scaling_factor=False,
    rm_self_attention=False,
):
    """Original GIDD self-correction, kept as an ablation helper."""
    logits = _call_model(
        model, z_t, t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
        rm_self_attention=rm_self_attention,
    ).clone()
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)
    z_tm1 = sample_categorical(p_t)

    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)
    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
    return z_tm1, acc


def correction_step(
    model,
    tokenizer,
    z_t,
    t,
    temp,
    tokens_per_step=1,
    latent_noise=False,
    activate_nvib_noise=False,
    use_trained_scaling_factor=False,
    rm_self_attention=False,
):
    """Targeted GIDD self-correction used between Psi-style re-noising moves.

    Position selection uses the raw-logit disagreement margin.  Temperature affects
    only the replacement distribution.  Unlike the user's previous version, MASK
    positions ARE editable: Psi-style forward re-noising can legitimately introduce
    a mask, so the following correction step must be able to denoise it again.
    """
    if temp <= 0:
        raise ValueError(f"Temperature must be > 0, got {temp}")

    logits = _call_model(
        model, z_t, t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
        rm_self_attention=rm_self_attention,
    ).clone()
    logits[..., tokenizer.mask_token_id] = -torch.inf

    pred_tokens = logits.argmax(dim=-1)
    self_acc = (z_t == pred_tokens).float().mean().item()

    best_logits = logits.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

    # If z_t contains a MASK introduced by the Psi re-noising step, its logit was
    # intentionally set to -inf.  That yields +inf disagreement margin and therefore
    # prioritizes denoising the mask, which is exactly what we want.
    current_logits = logits.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    margin = best_logits - current_logits

    correctable = pred_tokens != z_t
    if tokenizer.pad_token_id is not None:
        correctable = correctable & (z_t != tokenizer.pad_token_id)
    # DO NOT exclude mask positions here.

    score = margin.masked_fill(~correctable, -torch.inf)
    num_correctable = int(correctable.sum().item())

    if num_correctable == 0:
        return {
            "z_next": z_t.clone(),
            "self_acc": self_acc,
            "num_changed": 0,
            "num_correctable": 0,
            "has_correctable": False,
            "mean_margin": 0.0,
            "status": "fixed_point",
        }

    k = min(int(tokens_per_step), z_t.shape[-1], num_correctable)
    top_scores, ids = torch.topk(score, k=k, dim=-1)
    valid = torch.isfinite(top_scores) | torch.isposinf(top_scores)

    vocab_size = logits.shape[-1]
    selected_logits = logits.gather(
        1, ids.unsqueeze(-1).expand(-1, -1, vocab_size)
    ).clone()
    current_selected = z_t.gather(-1, ids)

    # Force a real edit at a position that the model says is wrong.
    selected_logits.scatter_(-1, current_selected.unsqueeze(-1), -torch.inf)

    selected_probs = torch.softmax(selected_logits.float() / temp, dim=-1)
    sampled_tokens = sample_categorical(selected_probs)
    sampled_tokens = torch.where(valid, sampled_tokens, current_selected)

    z_next = z_t.scatter(-1, ids, sampled_tokens)
    changed = valid & (sampled_tokens != current_selected)
    num_changed = int(changed.sum().item())

    finite_scores = top_scores[torch.isfinite(top_scores)]
    if finite_scores.numel() > 0:
        mean_margin = finite_scores.mean().item()
    elif valid.any():
        mean_margin = float("inf")
    else:
        mean_margin = 0.0

    return {
        "z_next": z_next,
        "self_acc": self_acc,
        "num_changed": num_changed,
        "num_correctable": num_correctable,
        "has_correctable": True,
        "mean_margin": mean_margin,
        "status": "changed" if num_changed > 0 else "no_op",
    }


def evaluate_self_accuracy(
    model,
    tokenizer,
    z_t,
    t,
    latent_noise=False,
    activate_nvib_noise=False,
    use_trained_scaling_factor=False,
    rm_self_attention=False,
):
    logits = _call_model(
        model, z_t, t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
        rm_self_attention=rm_self_attention,
    ).clone()
    logits[..., tokenizer.mask_token_id] = -torch.inf
    acc = (z_t == logits.argmax(-1)).float().mean().item()
    return logits, acc


def _noise_rate_at_t(noise_schedule, t_value: float, device) -> float:
    """Return total beta mass (= probability of taking the forward-noise branch)."""
    if not hasattr(noise_schedule, "get_alpha_betapi"):
        raise TypeError(
            "Psi post-hoc re-noising currently expects GIDD HybridDiffusion "
            "(noise_schedule.get_alpha_betapi is missing)."
        )
    t = torch.tensor([float(t_value)], device=device, dtype=torch.float32)
    alpha, beta_pi = noise_schedule.get_alpha_betapi(t)
    # Both are equivalent up to numerical precision.  beta sum is the clearest.
    return float(beta_pi.sum(-1).mean().item())


def find_noise_t_for_target_events(
    noise_schedule,
    seq_len: int,
    target_events: float,
    device,
    lo: float = 1e-7,
    hi: float = 0.25,
    iterations: int = 50,
) -> float:
    """Find t whose expected number of forward-noise branch events is target_events.

    This calibrates the new stochastic re-noising move to something directly
    comparable to the user's previous ``escape_num_tokens=8`` setting.
    Note that an event can redraw the same visible token, so the *net* number of
    changed tokens can be slightly smaller than target_events.
    """
    target_rate = min(max(float(target_events) / max(int(seq_len), 1), 0.0), 0.999)
    if target_rate <= 0:
        return lo

    if _noise_rate_at_t(noise_schedule, hi, device) < target_rate:
        raise ValueError(
            f"Target {target_events} noise events over length {seq_len} is too large "
            f"for search upper bound t={hi}. Increase ++psi_noise_t explicitly."
        )

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _noise_rate_at_t(noise_schedule, mid, device) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def psi_forward_renoise(
    noise_schedule,
    tokenizer,
    z_t: torch.Tensor,
    noise_t: float,
    max_retries: int = 8,
):
    """Sample a small perturbation from GIDD's exact hybrid forward kernel.

    For each position with current one-hot token x:
        q_t(z|x) = alpha_t * delta_x + beta_t * pi_t.

    We sample this efficiently without allocating a B x L x V one-hot tensor.
    The random positions and their replacement tokens therefore come from the
    *trained GIDD forward process*, not from a hand-written uniform heuristic.
    """
    if not hasattr(noise_schedule, "get_alpha_betapi"):
        raise TypeError(
            "psi_forward_renoise requires GIDD HybridDiffusion with get_alpha_betapi()."
        )

    batch_size, seq_len = z_t.shape
    device = z_t.device
    t_noise = torch.full(
        (batch_size,), float(noise_t), device=device, dtype=torch.float32
    )
    alpha, beta_pi = noise_schedule.get_alpha_betapi(t_noise)
    alpha = alpha.squeeze(-1).float()              # [B]
    beta_pi = beta_pi.float()                     # [B, V]
    noise_mass = beta_pi.sum(-1).clamp_min(1e-30) # [B]
    noise_dist = beta_pi / noise_mass[:, None]

    # Protect padding.  Everything else, including visible tokens, may be perturbed.
    editable = torch.ones_like(z_t, dtype=torch.bool)
    if tokenizer.pad_token_id is not None:
        editable &= z_t != tokenizer.pad_token_id

    last = None
    for attempt in range(max(1, int(max_retries))):
        take_noise = torch.rand(batch_size, seq_len, device=device) < noise_mass[:, None]
        take_noise &= editable

        noise_tokens = torch.empty_like(z_t)
        for b in range(batch_size):
            noise_tokens[b] = torch.multinomial(
                noise_dist[b],
                num_samples=seq_len,
                replacement=True,
            )

        z_next = torch.where(take_noise, noise_tokens, z_t)
        changed = editable & (z_next != z_t)
        event_count = int(take_noise.sum().item())
        num_changed = int(changed.sum().item())
        num_masks = int((changed & (z_next == tokenizer.mask_token_id)).sum().item())
        num_visible_random = num_changed - num_masks

        last = {
            "z_next": z_next,
            "num_events": event_count,
            "num_changed": num_changed,
            "num_masks": num_masks,
            "num_visible_random": num_visible_random,
            "noise_t": float(noise_t),
            "noise_rate": float(noise_mass.mean().item()),
            "attempt": attempt + 1,
            "changed_positions": torch.nonzero(changed, as_tuple=False),
        }
        if num_changed > 0:
            return last

    return last


def _print_psi_changes(sample_idx, escape_number, z_old, psi_out, tokenizer, max_print=12):
    positions = psi_out["changed_positions"]
    print(
        f"[sample {sample_idx}] PSI-RENOISE #{escape_number}: "
        f"t={psi_out['noise_t']:.7f}, forward-events={psi_out['num_events']}, "
        f"net-changes={psi_out['num_changed']}, masks={psi_out['num_masks']}, "
        f"visible-random={psi_out['num_visible_random']}"
    )
    if positions is None or positions.numel() == 0:
        return
    for row in positions[:max_print]:
        b, pos = int(row[0].item()), int(row[1].item())
        old_id = int(z_old[b, pos].item())
        new_id = int(psi_out["z_next"][b, pos].item())
        try:
            old_text = tokenizer.decode([old_id])
            new_text = tokenizer.decode([new_id])
        except Exception:
            old_text, new_text = str(old_id), str(new_id)
        print(f"    pos={pos:4d}: {old_text!r} -> {new_text!r}")
    if positions.shape[0] > max_print:
        print(f"    ... {positions.shape[0] - max_print} more changed positions")


@hydra.main(config_path="../configs", config_name="self_correction", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    latent_noise = bool(getattr(args, "latent_noise", False))
    activate_nvib_noise = bool(getattr(args, "activate_nvib_noise", False))
    use_trained_scaling_factor = bool(getattr(args, "use_trained_scaling_factor", False))
    rm_self_attention = bool(getattr(args, "rm_self_attention", False))

    print("args.latent_noise", latent_noise)
    print("args.activate_nvib_noise", activate_nvib_noise)
    print("args.use_trained_scaling_factor", use_trained_scaling_factor)
    print("args.rm_self_attention", rm_self_attention)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)
    model = torch.compile(model)

    model_kwargs = dict(
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
        rm_self_attention=rm_self_attention,
    )

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)
    if z_ts.ndim != 2:
        raise ValueError(f"Expected samples tensor [N, L], got shape={tuple(z_ts.shape)}")

    # -------------------------- Psi post-hoc settings --------------------------
    max_psi_escapes = int(getattr(args, "max_psi_escapes", 3))
    psi_trigger = str(getattr(args, "psi_trigger", "both"))
    if psi_trigger not in {"fixed_point", "stagnation", "both"}:
        raise ValueError("psi_trigger must be fixed_point, stagnation, or both")

    # If psi_noise_t <= 0, calibrate t to produce ~psi_target_events forward-noise
    # events per 512-token sequence.  This makes ++psi_target_events=8 directly
    # comparable to the user's previous ++escape_num_tokens=8.
    psi_noise_t = float(getattr(args, "psi_noise_t", -1.0))
    psi_target_events = float(getattr(args, "psi_target_events", 8.0))
    psi_max_retries = int(getattr(args, "psi_max_retries", 8))

    if psi_noise_t <= 0:
        psi_noise_t = find_noise_t_for_target_events(
            noise_schedule,
            seq_len=z_ts.shape[-1],
            target_events=psi_target_events,
            device=device,
        )

    psi_rate = _noise_rate_at_t(noise_schedule, psi_noise_t, device)
    expected_events = psi_rate * z_ts.shape[-1]

    print()
    print("=== PSI-INSPIRED POST-HOC RE-NOISING ===")
    print("IMPORTANT: exact Psi belongs in the generation sampler.")
    print("This script keeps your saved-PT self-correction workflow and uses")
    print("GIDD's exact forward kernel as a stochastic escape/rejuvenation move.")
    print("max_psi_escapes:", max_psi_escapes)
    print("psi_trigger:", psi_trigger)
    print("psi_noise_t:", psi_noise_t)
    print("forward-noise rate:", psi_rate)
    print("expected forward events / sample:", expected_events)
    print("psi_target_events:", psi_target_events)
    print("psi_max_retries:", psi_max_retries)
    print("========================================")
    print()

    metrics = []
    samples = []

    for sample_idx, z_t in enumerate(
        tqdm.tqdm(z_ts, desc="Psi self-correction", dynamic_ncols=True, smoothing=0.0)
    ):
        z_t = z_t.unsqueeze(0).to(device)
        z_t_init = z_t.clone()
        t = torch.full((z_t.shape[0],), device=device, fill_value=args.t0)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            _, init_acc = evaluate_self_accuracy(
                model=model, tokenizer=tokenizer, z_t=z_t, t=t, **model_kwargs
            )

        max_acc = init_acc
        curr_patience = 0
        converged = 0
        early_stopped = 0
        num_psi_escapes = 0
        total_psi_events = 0
        total_psi_changes = 0
        total_psi_masks = 0
        total_psi_visible_random = 0
        total_edit_steps = 0
        total_token_edits = 0

        def do_psi_escape(reason: str):
            nonlocal z_t, num_psi_escapes, total_psi_events, total_psi_changes
            nonlocal total_psi_masks, total_psi_visible_random, max_acc, curr_patience

            old = z_t.clone()
            psi_out = psi_forward_renoise(
                noise_schedule=noise_schedule,
                tokenizer=tokenizer,
                z_t=z_t,
                noise_t=psi_noise_t,
                max_retries=psi_max_retries,
            )
            if psi_out is None or psi_out["num_changed"] == 0:
                print(f"[sample {sample_idx}] Psi re-noising produced no net change ({reason}).")
                return False

            num_psi_escapes += 1
            total_psi_events += psi_out["num_events"]
            total_psi_changes += psi_out["num_changed"]
            total_psi_masks += psi_out["num_masks"]
            total_psi_visible_random += psi_out["num_visible_random"]
            _print_psi_changes(sample_idx, num_psi_escapes, old, psi_out, tokenizer)
            z_t = psi_out["z_next"]

            # A re-noising move intentionally leaves the previous fixed point / plateau,
            # so the old best self-accuracy should not instantly trigger early stopping.
            max_acc = -float("inf")
            curr_patience = 0
            return True

        for i in range(args.num_denoising_steps):
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                out = correction_step(
                    model=model,
                    tokenizer=tokenizer,
                    z_t=z_t,
                    t=t,
                    temp=args.temp,
                    tokens_per_step=args.tokens_per_step,
                    **model_kwargs,
                )

            acc = out["self_acc"]

            # ---- Exact fixed point of the current self-correction map ----
            if not out["has_correctable"]:
                print(f"\n[sample {sample_idx}] FIXED POINT at step {i} (self_acc={acc:.6f})")
                can_psi = psi_trigger in {"fixed_point", "both"} and num_psi_escapes < max_psi_escapes
                if can_psi and do_psi_escape("fixed_point"):
                    continue

                converged = 1
                if num_psi_escapes >= max_psi_escapes:
                    print(f"[sample {sample_idx}] Psi escape budget exhausted.")
                break

            # ---- Ordinary targeted self-correction edit ----
            if out["num_changed"] == 0:
                print(f"[sample {sample_idx}] NO-OP at step {i}; retrying.")
                continue

            z_t = out["z_next"]
            total_edit_steps += 1
            total_token_edits += out["num_changed"]

            # acc describes the pre-update state, matching the user's previous logic.
            if acc > max_acc:
                max_acc = acc
                curr_patience = 0
            else:
                curr_patience += 1

            # ---- Oscillation / stagnation escape ----
            if curr_patience > args.max_patience:
                can_psi = psi_trigger in {"stagnation", "both"} and num_psi_escapes < max_psi_escapes
                if can_psi and do_psi_escape("stagnation"):
                    print(f"[sample {sample_idx}] STAGNATION -> Psi re-noise at step {i}")
                    continue

                early_stopped = 1
                print(f"[sample {sample_idx}] EARLY STOP at step {i}")
                break

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            _, final_acc = evaluate_self_accuracy(
                model=model, tokenizer=tokenizer, z_t=z_t, t=t, **model_kwargs
            )

        num_changes = int((z_t_init != z_t).sum().item())
        samples.append(z_t)
        metrics.append({
            "sample_idx": sample_idx,
            "init_acc": init_acc,
            "final_acc": final_acc,
            "improvement": final_acc - init_acc,
            "num_changes": num_changes,
            "total_token_edits": total_token_edits,
            "total_edit_steps": total_edit_steps,
            "psi_escapes": num_psi_escapes,
            "psi_forward_events": total_psi_events,
            "psi_net_changes": total_psi_changes,
            "psi_masks": total_psi_masks,
            "psi_visible_random": total_psi_visible_random,
            "psi_noise_t": psi_noise_t,
            "converged": converged,
            "early_stopped": early_stopped,
        })

    samples = torch.cat(samples, dim=0).cpu()
    corrected_samples_path = hydra.utils.to_absolute_path(args.corrected_samples_path)
    print("\nSaving corrected samples:", corrected_samples_path)
    torch.save(samples, corrected_samples_path)

    df = pd.DataFrame(metrics)
    metrics_path = hydra.utils.to_absolute_path(args.metrics_path)
    df.to_csv(metrics_path, index=False)

    print()
    print(
        f"Results for {args.path} "
        f"(temp={args.temp}, max_patience={args.max_patience}, psi_noise_t={psi_noise_t:.7g})"
    )
    print(df.describe().to_markdown())
    print()
    print("Mean init acc:", df["init_acc"].mean())
    print("Mean final acc:", df["final_acc"].mean())
    print("Mean improvement:", df["improvement"].mean())
    print("Samples using Psi re-noise:", (df["psi_escapes"] > 0).sum())
    print("Total Psi re-noise calls:", df["psi_escapes"].sum())
    print("Total Psi forward events:", df["psi_forward_events"].sum())
    print("Total Psi net changes:", df["psi_net_changes"].sum())
    print("  masks:", df["psi_masks"].sum())
    print("  visible-random:", df["psi_visible_random"].sum())


if __name__ == "__main__":
    main()