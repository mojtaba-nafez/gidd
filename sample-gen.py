from gidd import GiddPipeline
import torch

# Download a pretrained model from HuggingFace
device = "cuda" if torch.cuda.is_available() else "cpu"

# pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
pipe = GiddPipeline.from_pretrained("./weights/gidd-base-pu-0.2", trust_remote_code=True)
pipe.to(device)

# Generate samples
texts = pipe.generate(num_samples=1, num_inference_steps=128)

# Run self-correction step
corrected_texts = pipe.self_correction(texts, num_inference_steps=128, early_stopping=True, temperature=0.1)

for i, (text, corrected_text) in enumerate(zip(texts, corrected_texts)):
    print(f"Sample {i+1}:")
    print("Original Text:")
    print(text)
    print("Corrected Text:")
    print(corrected_text)
    print("-" * 50)
