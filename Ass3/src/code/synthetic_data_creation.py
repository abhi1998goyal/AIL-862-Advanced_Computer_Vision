import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import shutil


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

classes = ["dog","cat","bird"]
num_classes = len(classes)
n_images_per_class = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

os.makedirs("synthetic_dataset", exist_ok=True)

prompt_variations = [
    "in a natural outdoor setting",
    "with vibrant colors",
    "captured in soft lighting",
    "with a shallow depth of field",
    "in a candid moment",
    "during golden hour",
    "with dramatic shadows",
    "with high contrast",
    "with a blurry background",
    "in a whimsical style",
    "with artistic flair",
    "in a studio setting",
    "with surreal colors",
    "in a minimalist composition",
    "with fine details",
    "in an urban environment",
    "with a vintage look",
    "in a lively scene",
    "with dynamic lighting",
    "in a serene mood"
]

def generate_synthetic_images():
    data = {cls: [] for cls in classes}
    for cls in classes:
        base_prompt = f"A high-quality cartoon illustration of a {cls}"
        print(f"Generating images for class: {cls}")
        for i in range(0,n_images_per_class):
       
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            
            variation = random.choice(prompt_variations)
            prompt = f"{base_prompt} {variation}"
            
            if device == "cuda":
                with torch.autocast("cuda"):
                    image = pipe(prompt, generator=generator).images[0]
            else:
                image = pipe(prompt, generator=generator).images[0]
                
            data[cls].append(image)
            image.save(f"synthetic_dataset/{cls}_{i}.png")
    return data

synthetic_data = generate_synthetic_images()


#shutil.make_archive("/kaggle/working/synthetic_dataset", 'zip', "synthetic_dataset")

