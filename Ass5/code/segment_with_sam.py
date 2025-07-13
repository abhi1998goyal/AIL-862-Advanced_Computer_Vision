import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor

# ======= CONFIGURATION =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT = "a photo of a dog"  # (For reference; not used directly here)
IMAGE_DIR = "/home/SCV/Ass5/processed_data/C_prime"  # Update to your images directory
MASK_SAVE_DIR = os.path.join(IMAGE_DIR, "masks")
SAM_CKPT = "/home/SCV/Ass5/saved_model/sam_vit_b_01ec64.pth"  # Update this to your SAM checkpoint path
os.makedirs(MASK_SAVE_DIR, exist_ok=True)
# =============================

# --- Step 1: Load CLIP Model and Processor ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Enable output of attentions from the vision model
clip_model.vision_model.config.output_attentions = True

# --- Step 2: Define the Preprocessing Pipeline ---
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                         std=(0.2686, 0.2613, 0.2758))
])

# --- Step 3: Load the Segment Anything Model (SAM) ---
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.to(DEVICE)
predictor = SamPredictor(sam)

# --- Step 4: Process Each Image ---
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

for img_name in tqdm(image_files):
    try:
        # Load the raw image and convert to RGB
        img_path = os.path.join(IMAGE_DIR, img_name)
        raw_image = Image.open(img_path).convert("RGB")
        
        # Preprocess image for CLIP (resize, normalize, and add batch dimension)
        image_tensor = clip_preprocess(raw_image).unsqueeze(0).to(DEVICE)
        
        # Obtain CLIP vision model outputs with attentions enabled.
        # The output is a CLIPVisionModelOutput with an "attentions" attribute.
        with torch.no_grad():
            outputs = clip_model.vision_model(pixel_values=image_tensor, output_attentions=True)
        
        # Extract attentions from the last transformer layer.
        # Each attention tensor has shape [B, num_heads, seq_len, seq_len].
        attentions = outputs.attentions[-1]
        
        # Average attention across heads.
        attn_map = attentions.mean(dim=1)  # Now shape is [B, seq_len, seq_len]
        
        # We want to get the attention that the CLS token (position 0) pays to each patch.
        # Exclude the CLS token itself by taking indices 1: (assume sequence length = 1 + num_patches).
        cls_attn = attn_map[0, 0, 1:]  # Shape: [seq_len - 1]
        
        # For CLIP ViT-B/32 on a 224x224 image, the number of patch tokens is usually 14x14 = 196.
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        if grid_size * grid_size != cls_attn.shape[0]:
            raise ValueError("Unexpected number of patch tokens found.")
        
        # Reshape the attention vector to a 2D spatial map.
        cls_attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
        
        # Normalize the attention map between 0 and 1.
        cls_attn_map = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min() + 1e-8)
        
        # Upsample the attention map to the original image size.
        # Note: raw_image.size returns (width, height)
        upsampled_attn = cv2.resize(cls_attn_map, raw_image.size)
        
        # Threshold the upsampled attention map to obtain a binary mask.
        # Here, we use 50% of the max value as the threshold.
        threshold = 0.5 * upsampled_attn.max()
        binary_mask = upsampled_attn > threshold
        
        # Compute bounding box coordinates from the binary mask.
        y_coords, x_coords = np.where(binary_mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            print(f"No salient area detected in {img_name}. Skipping.")
            continue
        
        box = np.array([x_coords.min()-20, y_coords.min()-20, x_coords.max()+20, y_coords.max()+20]).reshape(1, 4)
        
        # Provide the image and the bounding box to SAM to refine the segmentation.
        predictor.set_image(np.array(raw_image))
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        
        # Prepare and save the final mask.
        final_mask = (masks[0].astype(np.uint8)) * 255
        mask_save_path = os.path.join(MASK_SAVE_DIR, img_name.replace(".jpg", ".png"))
        cv2.imwrite(mask_save_path, final_mask)
    
    except Exception as e:
        print(f"Error on {img_name}: {e}")

print(f"\nDone. Saved masks in: {MASK_SAVE_DIR}")
