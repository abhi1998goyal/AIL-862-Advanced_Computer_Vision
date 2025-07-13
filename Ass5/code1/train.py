import os
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)

C_PRIME_DIR    = 'processed_data/C_prime'
C_PRIME_IMG    = os.path.join(C_PRIME_DIR)
C_PRIME_MASK   = os.path.join(C_PRIME_DIR, 'masks')
VAL_TEST_IMG   = 'processed_data/D/images'
VAL_TEST_MASK  = 'processed_data/D/treated_mask'
WORK_DIR       = 'processed_data/C_prime_val_train'
os.makedirs(WORK_DIR, exist_ok=True)

all_imgs = sorted([f for f in os.listdir(C_PRIME_IMG) if f.endswith('.jpg')])
train_ids, val_ids = train_test_split(all_imgs, test_size=0.2, random_state=42)

def make_split(dir_base, ids):
    img_dir = os.path.join(dir_base, 'images')
    mask_dir= os.path.join(dir_base, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for fname in ids:
        shutil.copy(os.path.join(C_PRIME_IMG, fname), img_dir)
        mask_name = fname.replace('.jpg', '.png')
        shutil.copy(os.path.join(C_PRIME_MASK, mask_name), mask_dir)
    return img_dir, mask_dir

train_img_dir, train_mask_dir = make_split(os.path.join(WORK_DIR,'train'), train_ids)
val_img_dir,   val_mask_dir   = make_split(os.path.join(WORK_DIR,'val'),   val_ids)

print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
        self.masks  = sorted([f for f in os.listdir(masks_dir)  if f.lower().endswith('.png')])
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir,  self.masks[idx])
        image     = Image.open(img_path).convert('RGB')
        mask      = Image.open(mask_path).convert('L')

        if self.img_transform:
            image = self.img_transform(image)
        else:
            image = T.ToTensor()(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = T.ToTensor()(mask)
        mask = (mask > 0).float()
        return image, mask

train_img_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_img_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
mask_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
])

batch_size = 8
train_loader = DataLoader(
    SegmentationDataset(train_img_dir, train_mask_dir, train_img_transform, mask_transform),
    batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    SegmentationDataset(val_img_dir, val_mask_dir, val_img_transform, mask_transform),
    batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    SegmentationDataset(VAL_TEST_IMG, VAL_TEST_MASK, val_img_transform, mask_transform),
    batch_size=batch_size, shuffle=False, num_workers=4
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=1
).to(device)
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_iou = 0.
epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.
    for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch} Train'):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()*imgs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_inter, val_union, _iou_count = 0., 0., 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f'Epoch {epoch} Val'):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))>0.5
            val_inter += (preds*masks).sum().item()
            val_union += ((preds+masks)>=1).sum().item()
            _iou_count += imgs.size(0)
    val_iou = val_inter/(val_union+1e-8)
    print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val IoU={val_iou:.4f}')
    if val_iou>best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(),'saved_model/best_on_val.pth')
        print('Saved best model on val')


model.load_state_dict(torch.load('saved_model/best_on_val.pth'))
test_inter, test_union = 0.,0.
with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc='Test Eval'):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = torch.sigmoid(model(imgs))>0.5
        test_inter += (preds*masks).sum().item()
        test_union += ((preds+masks)>=1).sum().item()
    test_iou = test_inter/(test_union+1e-8)
    print(f'Test IoU on D = {test_iou:.4f}')
