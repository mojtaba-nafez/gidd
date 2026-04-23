import torch
import hydra
from gidd.checkpoints import load_checkpoint  # or wherever load_checkpoint lives

# samples_path = "samples.pt"
# samples_path = "/idiap/temp/mnafez/research/gidd/corrected_samples.pt"
# samples_path = "/idiap/temp/mnafez/research/gidd/corrected_samples_original.pt"
# samples_path = "/idiap/temp/mnafez/research/gidd/corrected_samples_noisy_N1.pt"
# samples_path = "/idiap/temp/mnafez/research/gidd/corrected_samples_noisy_N2.pt"
# samples_path = "/idiap/temp/mnafez/research/gidd/samples_1024_original.pt"
samples_path = "/idiap/temp/mnafez/research/gidd/metrics-corrected_samples_noisy_N4.json"
ckpt_path = "/idiap/temp/mnafez/research/gidd/weights/gidd-base-pu-0.2"

device = torch.device("cpu")

# load tokenizer from the same checkpoint
model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)

# load token ids
samples = torch.load(samples_path, map_location="cpu")

print("samples shape:", samples.shape)

# detokenize
texts = tokenizer.batch_decode(samples, skip_special_tokens=True)

for i, txt in enumerate(texts):
    if i> (len(texts)-10):
        print(f"========== SAMPLE {i} ==========")
        print(txt)
    
