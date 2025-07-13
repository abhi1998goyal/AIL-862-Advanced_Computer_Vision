import os
import kagglehub

# Set credentials directly in code
os.environ["KAGGLE_USERNAME"] = "abgo24"
os.environ["KAGGLE_KEY"] = "a7b3a1f65653cfe31ab14253a0c72d97"

# Now download the dataset
path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

print("Dataset downloaded to:", path)
