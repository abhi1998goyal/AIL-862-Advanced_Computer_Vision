import shutil
import os

# Current dataset path
src = "/home/scai/phd/aiz218323/.cache/kagglehub/datasets/gopalbhattrai/pascal-voc-2012-dataset/versions/1"

# Desired destination
dst = "/home/scai/phd/aiz218323/scratch/abhishek_rl/scv/a5/data"

# Make sure destination exists (optional but safe)
os.makedirs(dst, exist_ok=True)

# Copy everything recursively
shutil.copytree(src, dst, dirs_exist_ok=True)

print("Dataset copied to:", dst)
