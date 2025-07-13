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
from torch.autograd import Function

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# I want: cat:0, dog:1, bird:2.
label_mapping = {0: 2, 1: 0, 2: 1}

class RemappedDataset(Dataset):
    def __init__(self, imagefolder, mapping):
        self.imagefolder = imagefolder
        self.mapping = mapping
    def __len__(self):
        return len(self.imagefolder)
    def __getitem__(self, idx):
        img, label = self.imagefolder[idx]
        remapped_label = self.mapping[label]
        return img, remapped_label

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

class DANNModel(nn.Module):
    def __init__(self, num_classes=3, domain_classes=2):
        super(DANNModel, self).__init__()
        base_model = models.resnet18(weights='DEFAULT')
        checkpoint = torch.load("saved_models/best_model_nreal.pth", map_location=device)
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("fc.")}
        base_model.load_state_dict(filtered_checkpoint, strict=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = base_model.fc.in_features 
        self.label_classifier = nn.Linear(self.feature_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 100),
            nn.ReLU(),
            nn.Linear(100, domain_classes)
        )
    def forward(self, x, lambda_=0.0):
        features = self.feature_extractor(x)
        features = features.view(-1, self.feature_dim)
        class_preds = self.label_classifier(features)
        reversed_features = grad_reverse(features, lambda_)
        domain_preds = self.domain_classifier(reversed_features)
        return class_preds, domain_preds

model_dann = DANNModel(num_classes=3, domain_classes=2).to(device)
optimizer_dann = optim.Adam(model_dann.parameters(), lr=1e-4)
domain_criterion = nn.CrossEntropyLoss()
class_criterion = nn.CrossEntropyLoss()

synthetic_dataset_raw = datasets.ImageFolder(root="synthetic", transform=transform)
synthetic_dataset = RemappedDataset(synthetic_dataset_raw, label_mapping)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=8, shuffle=True)

class FilteredRealDataset(Dataset):
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
        return img

filtered_root = "filtered_unlabeled_real"
filtered_real_dataset = FilteredRealDataset(filtered_root, transform=transform)
filtered_real_loader = DataLoader(filtered_real_dataset, batch_size=8, shuffle=True)

num_epochs = 10
lambda_domain = 1.0  

for epoch in range(num_epochs):
    model_dann.train()
    running_loss = 0.0
    clas_loss=0.0
    dom_loss=0.0
    syn_iter = iter(synthetic_loader)
    real_iter = iter(filtered_real_loader)
    num_batches = min(len(synthetic_loader), len(filtered_real_loader))
    
    for i in range(num_batches):
        try:
            syn_imgs, syn_labels = next(syn_iter)
        except StopIteration:
            syn_iter = iter(synthetic_loader)
            syn_imgs, syn_labels = next(syn_iter)
        try:
            real_imgs = next(real_iter)
        except StopIteration:
            real_iter = iter(filtered_real_loader)
            real_imgs = next(real_iter)
        
        syn_imgs = syn_imgs.to(device)
        syn_labels = syn_labels.to(device)
        real_imgs = real_imgs.to(device)
        
        domain_labels_syn = torch.zeros(syn_imgs.size(0), dtype=torch.long).to(device)
        domain_labels_real = torch.ones(real_imgs.size(0), dtype=torch.long).to(device)
        
        combined_imgs = torch.cat([syn_imgs, real_imgs], dim=0)
        combined_domain_labels = torch.cat([domain_labels_syn, domain_labels_real], dim=0)
        
        class_preds, domain_preds = model_dann(combined_imgs, lambda_=lambda_domain)
        
        loss_class = class_criterion(class_preds[:syn_imgs.size(0)], syn_labels)
        loss_domain = domain_criterion(domain_preds, combined_domain_labels)
        
        loss = loss_class + loss_domain
        
        optimizer_dann.zero_grad()
        loss.backward()
        optimizer_dann.step()
        
        running_loss += loss.item()
        clas_loss +=loss_class.item()
        dom_loss+=loss_domain.item()
    
    avg_loss = running_loss / num_batches
    avg_loss_cls = clas_loss/num_batches
    avg_dom_loss = dom_loss/num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, DOM_Loss: {avg_dom_loss:.4f}, CLS_Loss: {avg_loss_cls:.4f}")

os.makedirs("saved_models", exist_ok=True)
torch.save(model_dann.state_dict(), "saved_models/adapted_model.pth")
print("Adapted model saved to saved_models/adapted_model.pth")
