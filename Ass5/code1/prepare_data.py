import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
random.seed(42)
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import torch

device="cuda"
item_A_id = '2008_000336'
B="A photo of a dog"

voc_base = Path("data/VOC2012_train_val")
img_src  = f"{voc_base}/JPEGImages/{item_A_id}.jpg"
mask_src = f"{voc_base}/SegmentationClass/{item_A_id}.png"
A_dir = "processed_data/A"
os.makedirs(A_dir, exist_ok=True)
img_dst = os.path.join(A_dir, f"{item_A_id}.jpg")
mask_dst = os.path.join(A_dir, f"{item_A_id}.png")
shutil.copy(img_src, img_dst)
shutil.copy(mask_src, mask_dst)
print("Item A image and mask copied to A", A_dir)


main_dir = voc_base / "ImageSets" / "Main"
selected_classes = ["dog", "bus", "chair", "tvmonitor", "sofa"]
                    #,"aeroplane","bird","car","horse","sheep"]
class_to_images = defaultdict(list)
for cls in selected_classes:
    with open(main_dir / f"{cls}_train.txt", "r") as f:
        for line in f:
            img_id, label = line.strip().split()
            if label == "1" and img_id != item_A_id:
                class_to_images[cls].append(img_id)
all_image_ids = sorted(set(id for ids in class_to_images.values() for id in ids))
for cls in selected_classes:
    print(f"{cls:<7} → {len(class_to_images[cls])} images")
print(f"\n Total unique images across all selected classes: {len(all_image_ids)}")



dog_images = class_to_images["dog"]
non_dog_pool = sorted(
    set().union(*(class_to_images[c] for c in selected_classes if c != "dog"))
    - set(class_to_images["dog"])
)
dog_sample = random.sample(dog_images, 600)
non_dog_sample = random.sample(non_dog_pool, 200)
C_ids = sorted(dog_sample + non_dog_sample)
print(f"Total images in C: {len(C_ids)} (600 dog + 200 non-dog)")



voc_img_dir = voc_base / "JPEGImages"
dst_dir = Path("processed_data/C")
dst_dir.mkdir(parents=True, exist_ok=True)
for img_id in C_ids:
    src = voc_img_dir / f"{img_id}.jpg"
    dst = dst_dir / f"{img_id}.jpg"
    shutil.copy(src, dst)
print("All C images copied to:", dst_dir)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
prompt = B
text_inputs = processor(text=[prompt], return_tensors="pt").to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = F.normalize(text_features, dim=-1)
scores = []
C_dir = "processed_data/C"
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
top_k = 600
top_ids = [img_id for img_id, _ in scores[:top_k]]
print(f"Top {top_k} images selected for C′.")
C_prime_dir = "processed_data/C_prime"
os.makedirs(C_prime_dir, exist_ok=True)
for img_id in top_ids:
    src_path = os.path.join(C_dir, img_id)
    dst_path = os.path.join(C_prime_dir, img_id)
    shutil.copy(src_path, dst_path)
print(f" Copied {len(top_ids)} images to C′:", C_prime_dir)



dog_list_file = voc_base /"ImageSets/Main/dog_train.txt"
dog_id_map = {}
with open(dog_list_file, "r") as f:
    for line in f:
        img_id, label = line.strip().split()
        dog_id_map[img_id] = int(label)
C_prime_ids = [f[:-4] for f in os.listdir(C_prime_dir) if f.endswith(".jpg")]
dog_count = sum(1 for img_id in C_prime_ids if dog_id_map.get(img_id, -1) == 1)
non_dog_count = len(C_prime_ids) - dog_count
print(f"Total images in C′: {len(C_prime_ids)}")
print(f"Dog images: {dog_count}")
print(f"Non-dog images: {non_dog_count}")



val_ids_path = voc_base / "ImageSets/Segmentation/val.txt"
dog_val_path = voc_base / "ImageSets/Main/dog_val.txt"
with open(val_ids_path, "r") as f:
    val_ids = set(line.strip() for line in f)
dog_ids = []
with open(dog_val_path, "r") as f:
    for line in f:
        img_id, label = line.strip().split()
        if img_id in val_ids and label == "1":
            dog_ids.append(img_id)
print(f"Selected {len(dog_ids)} dog images for test set D")



D_dir = "processed_data/D"
os.makedirs(D_dir + "/images", exist_ok=True)
os.makedirs(D_dir + "/masks", exist_ok=True)
for img_id in dog_ids:
    img_src = voc_base / f"JPEGImages/{img_id}.jpg"
    mask_src = voc_base / f"SegmentationClass/{img_id}.png"
    
    shutil.copy(img_src, f"{D_dir}/images/{img_id}.jpg")
    shutil.copy(mask_src, f"{D_dir}/masks/{img_id}.png")
print(f"Copied {len(dog_ids)} images and masks to {D_dir}")



VOC_MASK_DIR   = "processed_data/D/masks"
OUTPUT_DIR_D     = "processed_data/D/treated_mask"
DOG_CLASS_ID   = 12  
os.makedirs(OUTPUT_DIR_D, exist_ok=True)
for fname in os.listdir(VOC_MASK_DIR):
    if not fname.endswith(".png"):
        continue
    mask_path = os.path.join(VOC_MASK_DIR, fname)
    mask_pil  = Image.open(mask_path)           
    mask_np   = np.array(mask_pil)             
    dog_mask = (mask_np == DOG_CLASS_ID).astype(np.uint8) * 255
    out_path = os.path.join(OUTPUT_DIR_D, fname)
    Image.fromarray(dog_mask).save(out_path)
print("Done! Dog masks saved to", OUTPUT_DIR_D)




IMAGE_DIR_C   = "processed_data/C_prime"               
OUTPUT_DIR_C  = "processed_data/C_prime/masks"       
SAM_CKPT    = "data/SAM/sam_vit_b_01ec64.pth"         
os.makedirs(OUTPUT_DIR_C, exist_ok=True)
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.to(device)
predictor = SamPredictor(sam)
for fname in tqdm(sorted(os.listdir(IMAGE_DIR_C))):
    if not fname.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(IMAGE_DIR_C, fname)
    image_pil = Image.open(img_path).convert("RGB")
    image_cv = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    h, w, _ = image_rgb.shape
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True
    )
    final_mask = (masks[0] * 255).astype(np.uint8) 
    mask_path = os.path.join(OUTPUT_DIR_C, fname.replace(".jpg",".png"))
    cv2.imwrite(mask_path, final_mask)
print(f"Done. Center‐point masks saved to '{OUTPUT_DIR_C}/'.")
