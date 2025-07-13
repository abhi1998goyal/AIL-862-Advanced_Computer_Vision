import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2

patch_tokens = None

def save_patch_tokens(module, input, output):
    global patch_tokens
    patch_tokens = output  

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# image_path = '/home/SCV/Ass4/data/1.png'
# image_cv = cv2.imread(image_path)

# image_blur = cv2.medianBlur(image_cv, 3)

# image_rgb = cv2.cvtColor(image_blur, cv2.COLOR_BGR2RGB)

# image= Image.fromarray(image_rgb)

k = 3
image_name="3.png"
image = Image.open(f'/home/SCV/Ass4/data/{image_name}').convert('RGB')

input_tensor = preprocess(image).unsqueeze(0)


model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
model.eval()

hook_handle = model.norm.register_forward_hook(save_patch_tokens)

with torch.no_grad():
    _ = model(input_tensor)

hook_handle.remove()

patch_features = patch_tokens[0, 1:, :]  
patch_features_np = patch_features.detach().cpu().numpy()

kmeans = KMeans(n_clusters=k, random_state=0).fit(patch_features_np)
cluster_labels = kmeans.labels_  
# num_patches = patch_features_np.shape[0]
segmentation_map = cluster_labels.reshape(14, 14)


segmentation_tensor = torch.tensor(segmentation_map).unsqueeze(0).unsqueeze(0).float()

plt.figure(figsize=(6,6))
plt.imshow(segmentation_tensor.squeeze().numpy(), cmap='jet')
plt.title("Segmentation Map using DINO Patch Features & Clustering")
plt.axis('off')
plt.savefig(f"/home/SCV/Ass4/k{k}/{image_name}", bbox_inches='tight', pad_inches=0)
#plt.show()
