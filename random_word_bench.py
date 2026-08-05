'''
python random_word_bench.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small"  batch_size=16 num_denoising_steps=128 temp=0.5  latent_noise=False activate_nvib_noise=False
''''
import numpy as np
from pathlib import Path
import torch
from gidd.models import dit
import hydra
from gidd.diffusion_process import get_noise_schedule
from gidd.modeling import get_tokenizer, get_model
from gidd.data import get_dataloaders
from gidd.loss import get_loss
from gidd.trainer import get_trainer
from gidd.optimizer import get_optimizer
from gidd.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)
import pandas as pd
import hydra
import tqdm
import torch

from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.utils import sample_categorical

@hydra.main(config_path="gidd/configs", config_name="self_correction", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    
    torch.set_grad_enabled(False)
    ckpt_path = hydra.utils.to_absolute_path(args.path)
    print("args.latent_noise", args.latent_noise)
    print("args.activate_nvib_noise", args.activate_nvib_noise)
    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)
    
    model = torch.compile(model)
    def print_num_parameters(model):
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    
    print_num_parameters(model)

    text = Path("/idiap/temp/mnafez/research/gidd/text_clean.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    chunk_size = 512
    vocab_size = tokenizer.vocab_size
    n_chunks = len(token_ids) // chunk_size
    print(
        f"Loaded {len(token_ids)} tokens "
        f"({n_chunks} chunks of {chunk_size})"
    )

    # with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
    #     x = torch.randint(
    #                 low=0,
    #                 high=vocab_size,
    #                 size=(1, 512),
    #                 device=device,
    #             )
    #     t = 0.01
    #     t = torch.full(
    #         (x.shape[0],),
    #         fill_value=t,
    #         device=device,
    #         dtype=torch.bfloat16,
    #     )
    #     logits = model(x, t, latent_noise=False, activate_nvib_noise=False)
    #     logits[..., tokenizer.mask_token_id] = -1e6
    

    t0 = 0.01
        

    
    total_correct = 0
    total_corrupted = 0
    total_copy_correct = 0
    total_clean_correct = 0
    total_num_clean = 0
    model.eval()

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size

            clean_tokens = torch.tensor(
                token_ids[start:end],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)  # [1, 1024]
            x = clean_tokens.clone()
            # -------------------------------------------------
            # Corrupt 20% of positions
            # -------------------------------------------------
            corruption_mask = (
                torch.rand_like(x.float()) < 0.20
            )

            random_tokens = torch.randint(
                low=0,
                high=vocab_size,
                size=x.shape,
                device=x.device,
            )
            
            x[corruption_mask] = random_tokens[corruption_mask]
            # x[corruption_mask] = 50257 # MDLM
            # x[corruption_mask] = 50258 # GIDD

            num_corrupted = corruption_mask.sum().item()
            num_clean = (~corruption_mask).sum().item()
            if num_corrupted == 0:
                continue

            t = torch.full(
                (x.shape[0],),
                fill_value=t0,
                device=device,
                dtype=torch.bfloat16,
            )
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=False).argmax(dim=-1)
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=False).argmax(dim=-1)
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=True).argmax(dim=-1)
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=True).argmax(dim=-1)
                logits = model(x, t, use_trained_scaling_factor=False, activate_nvib_noise=True)
                logits[..., tokenizer.mask_token_id] = -1e6
                pred_tokens = logits.argmax(dim=-1)

            
            correct = ((pred_tokens == clean_tokens) & corruption_mask).sum().item()
            copy_correct = ((x == pred_tokens) & corruption_mask).sum().item()
            clean_correct = ((clean_tokens == pred_tokens) & (~corruption_mask)).sum().item()
            

            '''
            corrupted_x = x[corruption_mask]
            corrupted_clean = clean_tokens[corruption_mask]
            corrupted_pred_tokens = pred_tokens[corruption_mask]
            print("\nFirst 20 corrupted positions:")
            for i in range(min(20, len(corrupted_x))):
                input_id = corrupted_x[i].item()
                target_id = corrupted_clean[i].item()
                pred_id = corrupted_pred_tokens[i].item()
                input_text = model.tokenizer.decode([input_id])
                target_text = model.tokenizer.decode([target_id])
                pred_text = model.tokenizer.decode([pred_id])
                print(
                    f"{i:02d}: "
                    f"input(x)={input_id:6d} ({repr(input_text)})  "
                    f"-> target={target_id:6d} ({repr(target_text)})  "
                    f"-> pred={pred_id:6d} ({repr(pred_text)})"
                )
            '''


            total_copy_correct += copy_correct
            total_correct += correct
            total_corrupted += num_corrupted

            total_clean_correct += clean_correct
            total_num_clean += num_clean

            if chunk_idx % 100 == 0:
                acc = total_correct / max(total_corrupted, 1)
                print(
                    f"Chunk {chunk_idx}/{n_chunks} "
                    f"Acc={acc:.4f}"
                )

    final_acc = 100 * total_correct / max(total_corrupted, 1)
    model_acc = 100 * total_correct / total_corrupted
    copy_acc = 100 * total_copy_correct / total_corrupted

    clean_acc = 100 * total_clean_correct / total_num_clean
    overall_acc = 100 * (total_correct+total_clean_correct) / (total_corrupted+total_num_clean)

    


    print("===============Final Results===============")
    print(
        f"Denoising Accuracy (Prediction == Ground Truth on Corrupted Tokens): "
        f"{model_acc:.6f} ({total_correct}/{total_corrupted})"
    )

    print(
        f"Copying Rate (Prediction == Corrupted Input on Corrupted Tokens): "
        f"{copy_acc:.6f} ({total_copy_correct}/{total_corrupted})"
    )
    print(
        f"Clean Acc (Prediction == Input on Clean Tokens): "
        f"{clean_acc:.6f} ({total_clean_correct}/{total_num_clean})"
    )

    print(
            f"Overall Acc (Prediction == Input): "
            f"{overall_acc:.6f} ({ (total_correct+total_clean_correct)}/{(total_corrupted+total_num_clean)})"
        )

  
    return final_acc

   
    

if __name__ == "__main__":
    main()