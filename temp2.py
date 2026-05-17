import numpy as np
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


@hydra.main(config_path="gidd/configs", config_name="gidd", version_base="1.1")
def main(config):
    dtype = parse_dtype(config.training.dtype)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(config)
    model = get_model(config, tokenizer, dtype=dtype)
    noise_schedule = get_noise_schedule(config, tokenizer)
    loss_fn = get_loss(config, tokenizer, noise_schedule)
    trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    trainer = trainer.to(device)

    print("trainer.device:", trainer.device)
    batch = {
        "input_ids": torch.randint(0, len(tokenizer), (4, 512), dtype=torch.long, device=trainer.device),
        "attention_mask": torch.ones((4, 512), dtype=torch.long, device=trainer.device),
    }
    trainer(batch, kl_loss=True)

    print("Model initialized successfully!")

if __name__ == "__main__":
    main()