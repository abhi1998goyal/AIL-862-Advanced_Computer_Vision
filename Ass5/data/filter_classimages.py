# dog_list_path = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/data/VOC2012_train_val/VOC2012_train_val/ImageSets/Main/dog_trainval.txt"

# with open(dog_list_path, "r") as f:
#     lines = f.readlines()

# # Keep only those images where class label is 1 (i.e., contains a dog)
# dog_images = [line.split()[0] for line in lines if line.split()[1] == "1"]

# print("Found", len(dog_images), "images with dogs.")
# print("First few image IDs:", dog_images[:20])
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm import tqdm
import os
import shutil

# Load model and processor
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Define your text prompt
# text_prompt = "a picture of a dog"

# # Image folder path
# C_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C"
# image_ids = [f for f in os.listdir(C_dir) if f.endswith(".jpg")]

# # Score each image using CLIP
# scores = []
# print("Scoring images using Hugging Face CLIP...")

# for image_id in tqdm(image_ids):
#     image_path = os.path.join(C_dir, image_id)
#     image = Image.open(image_path).convert("RGB")
    
#     # Prepare inputs for CLIP
#     inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True).to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image  # This gives raw similarity scores.
#         score = logits_per_image.softmax(dim=-1).squeeze().item()
#         scores.append((image_id, score))

# # Sort by similarity score
# scores.sort(key=lambda x: x[1], reverse=True)
# top_k = 300
# top_ids = [img_id for img_id, _ in scores[:top_k]]

# print(f"Top {top_k} images selected for C′.")

# C_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C"
# C_prime_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C_prime"

# # Ensure target directory exists
# os.makedirs(C_prime_dir, exist_ok=True)

# # Move images
# for img_id in top_ids:
#     src_path = os.path.join(C_dir, img_id)
#     dst_path = os.path.join(C_prime_dir, img_id)
#     shutil.copy(src_path, dst_path)

# print(f" Copied {len(top_ids)} images to C′:", C_prime_dir)

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode prompt
prompt = "a photo of a dog"
text_inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = F.normalize(text_features, dim=-1)

# Score each image
scores = []
C_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C"
image_ids = [f for f in os.listdir(C_dir) if f.endswith(".jpg")]

for img_id in tqdm(image_ids):
    img = Image.open(os.path.join(C_dir, img_id)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        img_features = model.get_image_features(**inputs)
        img_features = F.normalize(img_features, dim=-1)
        sim = torch.matmul(img_features, text_features.T).item()
    scores.append((img_id, sim))

scores.sort(key=lambda x: x[1], reverse=True)
top_k = 300
top_ids = [img_id for img_id, _ in scores[:top_k]]

print(f"Top {top_k} images selected for C′.")

C_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C"
C_prime_dir = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/processed_data/C_prime"

# Ensure target directory exists
os.makedirs(C_prime_dir, exist_ok=True)

# Move images
for img_id in top_ids:
    src_path = os.path.join(C_dir, img_id)
    dst_path = os.path.join(C_prime_dir, img_id)
    shutil.copy(src_path, dst_path)

print(f" Copied {len(top_ids)} images to C′:", C_prime_dir)