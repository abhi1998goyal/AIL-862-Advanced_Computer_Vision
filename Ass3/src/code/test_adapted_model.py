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

label_mapping = {0: 2, 1: 0, 2: 1}

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
        
       # base_model.load_state_dict(torch.load("saved_models/best_model_nreal.pth", map_location=device))
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

raw_test_dataset = datasets.ImageFolder(root=os.path.join("test", "non-synthetic"), transform=transform)
print("Raw test classes:", raw_test_dataset.classes) 
test_dataset = RemappedDataset(raw_test_dataset, label_mapping)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            class_preds, _ = model(imgs, lambda_=0.0)
            _, preds = torch.max(class_preds, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

model_dann = DANNModel(num_classes=3, domain_classes=2).to(device)
model_dann.load_state_dict(torch.load("saved_models/adapted_model.pth", map_location=device))

test_acc = evaluate_model(model_dann, test_loader)
print(f"\nFinal Test (Non-Synthetic) Accuracy: {test_acc*100:.2f}%")
