import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from torch.utils.data import Subset

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset_raw = datasets.ImageFolder(root="synthetic", transform=transform)
print("Raw training classes:", train_dataset_raw.classes) 

test_synthetic_raw = datasets.ImageFolder(root=os.path.join("test", "synthetic"), transform=transform)
test_non_synthetic_raw = datasets.ImageFolder(root=os.path.join("test", "non-synthetic"), transform=transform)
print("Raw test synthetic classes:", test_synthetic_raw.classes)
print("Raw test non-synthetic classes:", test_non_synthetic_raw.classes)


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

train_dataset = RemappedDataset(train_dataset_raw, label_mapping)
#train_dataset = Subset(train_dataset, list(range(00)))
test_synthetic_dataset = RemappedDataset(test_synthetic_raw, label_mapping)
test_non_synthetic_dataset = RemappedDataset(test_non_synthetic_raw, label_mapping)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(test_synthetic_dataset, batch_size=8, shuffle=False)  
non_syn_loader = DataLoader(test_non_synthetic_dataset, batch_size=8, shuffle=False)

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 3)  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 4  
patience = 3     
best_val_acc = 0.0
epochs_without_improve = 0

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

print("Starting training on synthetic data with early stopping...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
   
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Synthetic Val Acc: {val_acc*100:.2f}%")
    
  
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improve = 0
        
        torch.save(model.state_dict(), "saved_models/best_model_nreal.pth")
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break


model.load_state_dict(torch.load("saved_models/best_model_nreal.pth"))
non_syn_acc = evaluate(non_syn_loader)
print(f"\nFinal Non-Synthetic Test Accuracy: {non_syn_acc*100:.2f}%")
