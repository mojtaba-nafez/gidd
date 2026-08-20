import pandas as pd
import hydra
import tqdm
import torch

from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.utils import sample_categorical


# ============================================================
# ORIGINAL GIDD SELF-CORRECTION
# Kept only for comparison / ablation.
# ============================================================

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
):
    logits = model(
        z_t,
        t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
    )

    logits = logits.clone()
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)

    z_tm1 = sample_categorical(p_t)

    score = (
        (z_tm1 != z_t)
        * p_t.gather(
            -1,
            z_tm1.unsqueeze(-1),
        ).squeeze(-1)
    )

    ids = torch.topk(
        score,
        tokens_per_step,
        dim=-1,
    ).indices

    z_tm1 = z_t.scatter(
        -1,
        ids,
        z_tm1.gather(-1, ids),
    )

    acc = (
        z_tm1 == logits.argmax(-1)
    ).float().mean().item()

    return z_tm1, acc


# ============================================================
# MAIN TEMPERATURE-DECOUPLED CORRECTION STEP
# ============================================================

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
):
    """
    Temperature-decoupled self-correction.

    Position selection:
        Uses raw-logit disagreement:
            best_logit - current_token_logit

        Therefore temperature does NOT influence which position
        is selected.

    Replacement token:
        Sampled from temperature-scaled logits.

    Important:
        Once a position has been selected because the model
        prefers another token, the current token is removed from
        the replacement distribution. Therefore a stochastic
        no-op cannot be mistaken for convergence.
    """

    if temp <= 0:
        raise ValueError(
            f"Temperature must be > 0, got {temp}"
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(
        z_t,
        t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
    )

    logits = logits.clone()

    # Never generate MASK as a correction token
    logits[
        ...,
        tokenizer.mask_token_id
    ] = -torch.inf

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    pred_tokens = logits.argmax(
        dim=-1
    )

    self_acc = (
        z_t == pred_tokens
    ).float().mean().item()

    # --------------------------------------------------------
    # Raw-logit disagreement score
    # --------------------------------------------------------

    best_logits = logits.gather(
        -1,
        pred_tokens.unsqueeze(-1),
    ).squeeze(-1)

    current_logits = logits.gather(
        -1,
        z_t.unsqueeze(-1),
    ).squeeze(-1)

    margin = (
        best_logits
        - current_logits
    )

    # Model explicitly prefers another token
    correctable = (
        pred_tokens != z_t
    )

    # Do not change PAD
    if tokenizer.pad_token_id is not None:
        correctable = (
            correctable
            & (
                z_t
                != tokenizer.pad_token_id
            )
        )

    # Do not treat MASK positions as ordinary correction
    correctable = (
        correctable
        & (
            z_t
            != tokenizer.mask_token_id
        )
    )

    score = margin.masked_fill(
        ~correctable,
        -torch.inf,
    )

    num_correctable = (
        correctable.sum().item()
    )

    # --------------------------------------------------------
    # TRUE fixed point
    #
    # No token exists for which model argmax differs from
    # current token.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Select positions using raw-logit disagreement
    # --------------------------------------------------------

    k = min(
        int(tokens_per_step),
        z_t.shape[-1],
        int(num_correctable),
    )

    top_scores, ids = torch.topk(
        score,
        k=k,
        dim=-1,
    )

    valid = torch.isfinite(
        top_scores
    )

    # --------------------------------------------------------
    # Extract selected-position logits
    # --------------------------------------------------------

    vocab_size = logits.shape[-1]

    selected_logits = logits.gather(
        1,
        ids.unsqueeze(-1).expand(
            -1,
            -1,
            vocab_size,
        ),
    ).clone()

    current_selected = z_t.gather(
        -1,
        ids,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Position is already known to be inconsistent with
    # model argmax.
    #
    # Therefore prevent re-sampling exactly the current token.
    # --------------------------------------------------------

    selected_logits.scatter_(
        -1,
        current_selected.unsqueeze(-1),
        -torch.inf,
    )

    # --------------------------------------------------------
    # Temperature affects only replacement-token sampling
    # --------------------------------------------------------

    selected_probs = torch.softmax(
        selected_logits.float() / temp,
        dim=-1,
    )

    sampled_tokens = sample_categorical(
        selected_probs
    )

    # Invalid top-k entries remain unchanged
    sampled_tokens = torch.where(
        valid,
        sampled_tokens,
        current_selected,
    )

    # --------------------------------------------------------
    # Apply correction
    # --------------------------------------------------------

    z_next = z_t.scatter(
        -1,
        ids,
        sampled_tokens,
    )

    changed = (
        valid
        & (
            sampled_tokens
            != current_selected
        )
    )

    num_changed = (
        changed.sum().item()
    )

    valid_scores = (
        top_scores[valid]
    )

    mean_margin = (
        valid_scores.mean().item()
        if valid_scores.numel() > 0
        else 0.0
    )

    return {
        "z_next": z_next,
        "self_acc": self_acc,
        "num_changed": num_changed,
        "num_correctable": num_correctable,
        "has_correctable": True,
        "mean_margin": mean_margin,
        "status": (
            "changed"
            if num_changed > 0
            else "no_op"
        ),
    }


# ============================================================
# FIXED-POINT ESCAPE
# ============================================================

def escape_fixed_point(
    logits,
    tokenizer,
    z_t,
    num_escape_tokens=1,
    escape_temp=0.0,
):
    """
    Escape a true GIDD fixed point.

    At a fixed point:

        z_i == argmax model(z)_i

    for every editable position.

    Therefore normal self-correction has no candidate.

    We find the LEAST STABLE positions:

        stability_i =
            logit(current_i)
            - logit(best alternative_i)

    Small stability means the second-best token is close to
    the current one.

    We then replace a tiny number of those positions and let
    ordinary self-correction resume.

    escape_temp <= 0:
        deterministic second-best token.

    escape_temp > 0:
        sample from alternatives after excluding current token.
    """

    logits = logits.clone()

    logits[
        ...,
        tokenizer.mask_token_id
    ] = -torch.inf

    batch_size, seq_len, vocab_size = (
        logits.shape
    )

    # --------------------------------------------------------
    # Logit of current token
    # --------------------------------------------------------

    current_logits = logits.gather(
        -1,
        z_t.unsqueeze(-1),
    ).squeeze(-1)

    # --------------------------------------------------------
    # Remove current token to reveal best alternative
    # --------------------------------------------------------

    alternative_logits = (
        logits.clone()
    )

    alternative_logits.scatter_(
        -1,
        z_t.unsqueeze(-1),
        -torch.inf,
    )

    # --------------------------------------------------------
    # Strongest alternative
    # --------------------------------------------------------

    (
        best_alt_logits,
        best_alt_tokens,
    ) = alternative_logits.max(
        dim=-1
    )

    # --------------------------------------------------------
    # Stability margin
    #
    # Small = easier / safer position to perturb
    # --------------------------------------------------------

    stability_margin = (
        current_logits
        - best_alt_logits
    )

    # --------------------------------------------------------
    # Valid positions
    # --------------------------------------------------------

    valid_position = (
        z_t
        != tokenizer.mask_token_id
    )

    if tokenizer.pad_token_id is not None:
        valid_position = (
            valid_position
            & (
                z_t
                != tokenizer.pad_token_id
            )
        )

    stability_margin = (
        stability_margin.masked_fill(
            ~valid_position,
            torch.inf,
        )
    )

    num_valid = (
        valid_position.sum().item()
    )

    if num_valid == 0:

        return {
            "z_next": z_t.clone(),
            "escape_positions": None,
            "escape_margin": float("inf"),
            "num_changed": 0,
        }

    # --------------------------------------------------------
    # Select least-stable positions
    # --------------------------------------------------------

    k = min(
        int(num_escape_tokens),
        seq_len,
        int(num_valid),
    )

    escape_margins, ids = torch.topk(
        stability_margin,
        k=k,
        dim=-1,
        largest=False,
    )

    valid_escape = torch.isfinite(
        escape_margins
    )

    current_selected = z_t.gather(
        -1,
        ids,
    )

    # --------------------------------------------------------
    # Choose alternative
    # --------------------------------------------------------

    if escape_temp <= 0:

        # Deterministic strongest alternative
        replacement = (
            best_alt_tokens.gather(
                -1,
                ids,
            )
        )

    else:

        selected_alt_logits = (
            alternative_logits.gather(
                1,
                ids.unsqueeze(-1).expand(
                    -1,
                    -1,
                    vocab_size,
                ),
            )
        )

        selected_probs = torch.softmax(
            selected_alt_logits.float()
            / escape_temp,
            dim=-1,
        )

        replacement = sample_categorical(
            selected_probs
        )

    replacement = torch.where(
        valid_escape,
        replacement,
        current_selected,
    )

    # --------------------------------------------------------
    # Apply escape
    # --------------------------------------------------------

    z_next = z_t.scatter(
        -1,
        ids,
        replacement,
    )

    changed = (
        replacement
        != current_selected
    ) & valid_escape

    num_changed = (
        changed.sum().item()
    )

    mean_escape_margin = (
        escape_margins[
            valid_escape
        ].mean().item()
        if valid_escape.any()
        else float("inf")
    )

    return {
        "z_next": z_next,
        "escape_positions": ids,
        "escape_margin": mean_escape_margin,
        "num_changed": num_changed,
    }


# ============================================================
# FORWARD / ACCURACY HELPER
# ============================================================

def evaluate_self_accuracy(
    model,
    tokenizer,
    z_t,
    t,
    latent_noise=False,
    activate_nvib_noise=False,
    use_trained_scaling_factor=False,
):
    """
    Evaluate self-accuracy using exactly the same model options
    as the correction step.
    """

    logits = model(
        z_t,
        t,
        latent_noise=latent_noise,
        activate_nvib_noise=activate_nvib_noise,
        use_trained_scaling_factor=use_trained_scaling_factor,
    )

    logits = logits.clone()

    logits[
        ...,
        tokenizer.mask_token_id
    ] = -torch.inf

    acc = (
        z_t == logits.argmax(-1)
    ).float().mean().item()

    return logits, acc


# ============================================================
# MAIN
# ============================================================

@hydra.main(
    config_path="../configs",
    config_name="self_correction",
    version_base="1.1",
)
def main(args):

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    torch.set_grad_enabled(
        False
    )

    # ========================================================
    # Load checkpoint
    # ========================================================

    ckpt_path = (
        hydra.utils.to_absolute_path(
            args.path
        )
    )

    print(
        "args.latent_noise",
        args.latent_noise,
    )

    print(
        "args.activate_nvib_noise",
        args.activate_nvib_noise,
    )

    print(
        "args.use_trained_scaling_factor",
        args.use_trained_scaling_factor,
    )

    (
        model,
        noise_schedule,
        tokenizer,
        config,
    ) = load_checkpoint(
        ckpt_path,
        device=device,
    )

    model.eval()

    config.training.eval_batch_size = (
        args.batch_size
    )

    dtype = parse_dtype(
        config.training.dtype
    )

    # ========================================================
    # Compile
    # ========================================================

    model = torch.compile(
        model
    )

    # ========================================================
    # Load generated samples
    # ========================================================

    samples_path = (
        hydra.utils.to_absolute_path(
            args.samples_path
        )
    )

    # Load only to preserve the original [num_samples, seq_len] shape
    reference_z_ts = torch.load(
        samples_path,
        weights_only=True,
    )

    num_samples, seq_len = reference_z_ts.shape

    # ------------------------------------------------------------
    # Create RANDOM token sequences from tokenizer vocabulary
    # ------------------------------------------------------------

    # Exclude special tokens such as:
    # MASK, PAD, BOS, EOS, etc.
    special_ids = set(tokenizer.all_special_ids)

    valid_token_ids = torch.tensor(
        [
            token_id
            for token_id in range(len(tokenizer))
            if token_id not in special_ids
        ],
        dtype=torch.long,
    )

    # Reproducible random generation
    generator = torch.Generator()
    generator.manual_seed(1234)

    random_indices = torch.randint(
        low=0,
        high=len(valid_token_ids),
        size=(num_samples, seq_len),
        generator=generator,
    )

    z_ts = valid_token_ids[random_indices]

    print("=== RANDOM START ===")
    print("z_ts.shape:", z_ts.shape)
    print("vocab_size:", len(tokenizer))
    print("valid_vocab_size:", len(valid_token_ids))
    print("excluded special IDs:", sorted(special_ids))
    print("====================")

    # ========================================================
    # OPTIONAL DEBUG:
    #
    # To test ONLY sample index 32:
    #
    # z_ts = z_ts[32:33, :]
    #
    # Comment this line for full evaluation.
    # ========================================================

    # z_ts = z_ts[32:33, :]

    # ========================================================
    # Fixed-point escape parameters
    #
    # These can be overridden using Hydra:
    #
    # ++max_fixed_point_escapes=3
    # ++escape_num_tokens=1
    # ++escape_temp=0.0
    # ========================================================

    max_fixed_point_escapes = int(
        getattr(
            args,
            "max_fixed_point_escapes",
            3,
        )
    )

    escape_num_tokens = int(
        getattr(
            args,
            "escape_num_tokens",
            1,
        )
    )

    escape_temp = float(
        getattr(
            args,
            "escape_temp",
            0.0,
        )
    )

    print()
    print(
        "=== FIXED-POINT ESCAPE SETTINGS ==="
    )

    print(
        "max_fixed_point_escapes:",
        max_fixed_point_escapes,
    )

    print(
        "escape_num_tokens:",
        escape_num_tokens,
    )

    print(
        "escape_temp:",
        escape_temp,
    )

    print(
        "==================================="
    )
    print()

    # ========================================================
    # Output storage
    # ========================================================

    metrics = []
    samples = []

    # ========================================================
    # Process samples one by one
    # ========================================================

    for sample_idx, z_t in enumerate(
        tqdm.tqdm(
            z_ts,
            desc="Correction",
            dynamic_ncols=True,
            smoothing=0.0,
        )
    ):

        # ----------------------------------------------------
        # Prepare sample
        # ----------------------------------------------------

        z_t = (
            z_t
            .unsqueeze(0)
            .to(device)
        )

        z_t_init = (
            z_t.clone()
        )

        t = torch.full(
            (z_t.shape[0],),
            device=device,
            fill_value=args.t0,
        )

        # ----------------------------------------------------
        # Initial self-accuracy
        #
        # IMPORTANT:
        # same model configuration as correction step.
        # ----------------------------------------------------

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
            ),
        ):

            (
                init_logits,
                init_acc,
            ) = evaluate_self_accuracy(
                model=model,
                tokenizer=tokenizer,
                z_t=z_t,
                t=t,
                latent_noise=args.latent_noise,
                activate_nvib_noise=args.activate_nvib_noise,
                use_trained_scaling_factor=(
                    args.use_trained_scaling_factor
                ),
            )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

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

        # ====================================================
        # Correction iterations
        # ====================================================

        for i in range(
            args.num_denoising_steps
        ):

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                ),
            ):

                out = correction_step(
                    model=model,
                    tokenizer=tokenizer,
                    z_t=z_t,
                    t=t,
                    temp=args.temp,
                    tokens_per_step=(
                        args.tokens_per_step
                    ),
                    latent_noise=(
                        args.latent_noise
                    ),
                    activate_nvib_noise=(
                        args.activate_nvib_noise
                    ),
                    use_trained_scaling_factor=(
                        args.use_trained_scaling_factor
                    ),
                )

            z_t_next = out[
                "z_next"
            ]

            acc = out[
                "self_acc"
            ]

            last_acc = acc

            # =================================================
            # TRUE FIXED POINT
            # =================================================

            if not out[
                "has_correctable"
            ]:

                print()
                print(
                    f"[sample {sample_idx}] "
                    f"FIXED POINT at step {i} "
                    f"(self_acc={acc:.6f})"
                )

                # ---------------------------------------------
                # Escape if budget remains
                # ---------------------------------------------

                if (
                    num_fixed_point_escapes
                    <
                    max_fixed_point_escapes
                ):

                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type,
                            dtype=dtype,
                        ),
                    ):

                        escape_logits = model(
                            z_t,
                            t,
                            latent_noise=(
                                args.latent_noise
                            ),
                            activate_nvib_noise=(
                                args.activate_nvib_noise
                            ),
                            use_trained_scaling_factor=(
                                args.use_trained_scaling_factor
                            ),
                        )

                    escape_out = (
                        escape_fixed_point(
                            logits=escape_logits,
                            tokenizer=tokenizer,
                            z_t=z_t,
                            num_escape_tokens=(
                                escape_num_tokens
                            ),
                            escape_temp=(
                                escape_temp
                            ),
                        )
                    )

                    # -----------------------------------------
                    # Escape failed
                    # -----------------------------------------

                    if (
                        escape_out[
                            "num_changed"
                        ]
                        == 0
                    ):

                        print(
                            f"[sample {sample_idx}] "
                            "Fixed-point escape failed."
                        )

                        converged = 1
                        break

                    # -----------------------------------------
                    # Print changed tokens
                    # -----------------------------------------

                    positions = (
                        escape_out[
                            "escape_positions"
                        ]
                    )

                    if positions is not None:

                        for j in range(
                            positions.shape[-1]
                        ):

                            pos = (
                                positions[
                                    0,
                                    j
                                ].item()
                            )

                            old_id = (
                                z_t[
                                    0,
                                    pos
                                ].item()
                            )

                            new_id = (
                                escape_out[
                                    "z_next"
                                ][
                                    0,
                                    pos
                                ].item()
                            )

                            try:

                                old_text = (
                                    tokenizer.decode(
                                        [old_id]
                                    )
                                )

                                new_text = (
                                    tokenizer.decode(
                                        [new_id]
                                    )
                                )

                            except Exception:

                                old_text = str(
                                    old_id
                                )

                                new_text = str(
                                    new_id
                                )

                            print(
                                f"[sample {sample_idx}] "
                                f"ESCAPE #{num_fixed_point_escapes + 1}: "
                                f"position={pos}, "
                                f"old={old_text!r}, "
                                f"new={new_text!r}, "
                                f"margin="
                                f"{escape_out['escape_margin']:.6f}"
                            )

                    # -----------------------------------------
                    # Accept escape
                    # -----------------------------------------

                    z_t = (
                        escape_out[
                            "z_next"
                        ]
                    )

                    num_fixed_point_escapes += 1

                    total_escape_changes += (
                        escape_out[
                            "num_changed"
                        ]
                    )

                    escape_margins.append(
                        escape_out[
                            "escape_margin"
                        ]
                    )

                    # -----------------------------------------
                    # CRITICAL:
                    #
                    # The previous fixed point may have
                    # self_acc=1.0.
                    #
                    # If max_acc remains 1.0, post-escape
                    # states can never "improve", causing
                    # artificial early stopping.
                    #
                    # Reset the local patience reference.
                    # -----------------------------------------

                    max_acc = -float(
                        "inf"
                    )

                    curr_patience = 0

                    # Let normal correction resume
                    continue

                # ---------------------------------------------
                # Escape budget exhausted
                # ---------------------------------------------

                print(
                    f"[sample {sample_idx}] "
                    "Fixed-point escape budget exhausted."
                )

                converged = 1
                break

            # =================================================
            # STOCHASTIC NO-OP
            #
            # This should now be extremely rare because the
            # current token is removed at selected positions.
            # =================================================

            if out[
                "num_changed"
            ] == 0:

                print(
                    f"[sample {sample_idx}] "
                    f"NO-OP at step {i}; retrying."
                )

                continue

            # =================================================
            # Normal correction happened
            # =================================================

            z_t = z_t_next

            total_edit_steps += 1

            total_token_edits += (
                out[
                    "num_changed"
                ]
            )

            # =================================================
            # Patience logic
            #
            # acc belongs to the pre-update state.
            # The updated state is evaluated on the next loop.
            # =================================================

            if (
                acc
                >
                max_acc
            ):

                max_acc = acc

                curr_patience = 0

            else:

                curr_patience += 1

                if (
                    curr_patience
                    >
                    args.max_patience
                ):

                    early_stopped = 1

                    print(
                        f"[sample {sample_idx}] "
                        f"EARLY STOP at step {i}"
                    )

                    break

        # ====================================================
        # Evaluate FINAL state
        #
        # Important because last_acc may correspond to the
        # state before the final modification.
        # ====================================================

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=dtype,
            ),
        ):

            (
                final_logits,
                final_acc,
            ) = evaluate_self_accuracy(
                model=model,
                tokenizer=tokenizer,
                z_t=z_t,
                t=t,
                latent_noise=(
                    args.latent_noise
                ),
                activate_nvib_noise=(
                    args.activate_nvib_noise
                ),
                use_trained_scaling_factor=(
                    args.use_trained_scaling_factor
                ),
            )

        # ====================================================
        # Net number of changed tokens
        # ====================================================

        num_changes = (
            z_t_init
            != z_t
        ).sum().item()

        # ====================================================
        # Save sample
        # ====================================================

        samples.append(
            z_t
        )

        # ====================================================
        # Save metrics
        # ====================================================

        metrics.append(
            {
                "sample_idx": sample_idx,

                "init_acc": init_acc,
                "final_acc": final_acc,

                "improvement": (
                    final_acc
                    - init_acc
                ),

                # Net final difference
                "num_changes": num_changes,

                # Total normal edits across trajectory
                "total_token_edits": (
                    total_token_edits
                ),

                "total_edit_steps": (
                    total_edit_steps
                ),

                # Fixed-point information
                "fixed_point_escapes": (
                    num_fixed_point_escapes
                ),

                "escape_token_changes": (
                    total_escape_changes
                ),

                "mean_escape_margin": (
                    sum(escape_margins)
                    / len(escape_margins)
                    if escape_margins
                    else 0.0
                ),

                "converged": converged,
                "early_stopped": early_stopped,
            }
        )

    # ========================================================
    # Save corrected samples
    # ========================================================

    samples = torch.cat(
        samples,
        dim=0,
    ).cpu()

    corrected_samples_path = (
        hydra.utils.to_absolute_path(
            args.corrected_samples_path
        )
    )

    print()
    print(
        "Saving corrected samples:",
        corrected_samples_path,
    )

    torch.save(
        samples,
        corrected_samples_path,
    )

    # ========================================================
    # Save metrics CSV
    # ========================================================

    df = pd.DataFrame(
        metrics
    )

    metrics_path = (
        hydra.utils.to_absolute_path(
            args.metrics_path
        )
    )

    df.to_csv(
        metrics_path,
        index=False,
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print(
        f"Results for {args.path} "
        f"(temp={args.temp}, "
        f"max_patience={args.max_patience})"
    )

    print()

    print(
        df.describe().to_markdown()
    )

    print()
    print(
        "Mean init acc:",
        df["init_acc"].mean(),
    )

    print(
        "Mean final acc:",
        df["final_acc"].mean(),
    )

    print(
        "Mean improvement:",
        df["improvement"].mean(),
    )

    print(
        "Samples requiring fixed-point escape:",
        (
            df["fixed_point_escapes"]
            > 0
        ).sum(),
    )

    print(
        "Total fixed-point escapes:",
        df[
            "fixed_point_escapes"
        ].sum(),
    )

    print(
        "Total escape token changes:",
        df[
            "escape_token_changes"
        ].sum(),
    )


if __name__ == "__main__":
    main()

'''


'''