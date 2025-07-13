import os
import random
import shutil
from PIL import Image
from torchvision.datasets import OxfordIIITPet

output_dir = "val/non-synthetic"
cat_dir = os.path.join(output_dir, "cat")
dog_dir = os.path.join(output_dir, "dog")
bird_dir = os.path.join(output_dir, "bird")

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)
os.makedirs(bird_dir, exist_ok=True)

pet_dataset = OxfordIIITPet(root='./data/oxford-iiit-pet', download=True)

cat_count = 0
dog_count = 0
max_images = 100  

for idx, (img, target) in enumerate(pet_dataset):
    try:
        label = target['label_cat_dog']
    except (TypeError, KeyError):
        label = 0 if target < 12 else 1

    if label == 0 and cat_count < max_images:
        filename = os.path.join(cat_dir, f"cat_{cat_count}.jpg")
        img.save(filename)
        cat_count += 1
    elif label == 1 and dog_count < max_images:
        filename = os.path.join(dog_dir, f"dog_{dog_count}.jpg")
        img.save(filename)
        dog_count += 1

    if cat_count >= max_images and dog_count >= max_images:
        break

print(f"Saved {cat_count} cat images and {dog_count} dog images.")



#cub_root = "/home/SCV/Ass3/data/CUB_200_2011/images"
cub_root = "/mnt/d/CV_dataset/data/CUB_200_2011/images"

bird_count = 0

for subfolder in os.listdir(cub_root):
    subfolder_path = os.path.join(cub_root, subfolder)
    if os.path.isdir(subfolder_path):
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
        num_to_sample = min(1, len(image_files))
        sampled_files = random.sample(image_files, num_to_sample)
        for file in sampled_files:
            src = os.path.join(subfolder_path, file)
            dst = os.path.join(bird_dir, f"bird_{bird_count}.jpg")
            try:
                shutil.copy(src, dst)
                bird_count += 1
            except Exception as e:
                print(f"Error copying {src}: {e}")

print(f"Saved {bird_count} bird images.")
