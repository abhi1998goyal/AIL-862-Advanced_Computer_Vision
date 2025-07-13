import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models, datasets
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) 
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

unlabeled_root = "unlabeled_data_real"  
unlabeled_dataset = UnlabeledDataset(unlabeled_root, transform=transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)


synthetic_model = models.resnet18(weights='DEFAULT')
synthetic_model.fc = nn.Linear(synthetic_model.fc.in_features, 3)
synthetic_model = synthetic_model.to(device)
synthetic_model.load_state_dict(torch.load("saved_models/best_model_nreal.pth", map_location=device))
synthetic_model.eval()

confidence_threshold = 0.95

filtered_paths = []         
filtered_pseudo_labels = [] 

with torch.no_grad():
    for imgs, paths in unlabeled_loader:
        imgs = imgs.to(device)
        outputs = synthetic_model(imgs)
        probs = F.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        for i in range(len(paths)):
            if max_probs[i].item() >= confidence_threshold:
                filtered_paths.append(paths[i])
                filtered_pseudo_labels.append(preds[i].item())

print(f"Filtered {len(filtered_paths)} images from unlabeled data based on confidence.")

filtered_root = "filtered_unlabeled_real"
os.makedirs(filtered_root, exist_ok=True)
for i, path in enumerate(filtered_paths):
    filename = os.path.basename(path)
    #new_filename = f"{filtered_pseudo_labels[i]}_{filename}"  
    shutil.copy(path, os.path.join(filtered_root, filename))
