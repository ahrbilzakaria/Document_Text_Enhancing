import os
import cv2
import glob
from Upscale_To_Enhance_text import upscale_images
from Enhance_contrast import adaptive_clahe
from Adaptive_unsharp_mask import adaptive_sharpen

# Define input and output directories
INPUT_DIR = "start"
UPSCALE_OUTPUT_DIR = "output"
CLAHE_OUTPUT_DIR = "clahe_output"
SHARPEN_OUTPUT_DIR = "sharpen_output"

# Ensure output directories exist
os.makedirs(UPSCALE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CLAHE_OUTPUT_DIR, exist_ok=True)
os.makedirs(SHARPEN_OUTPUT_DIR, exist_ok=True)

# Step 1: Upscale images
print("\n🔹 Upscaling images...")
upscale_images(input_dir=INPUT_DIR, output_dir=UPSCALE_OUTPUT_DIR, outscale=2, denoise_strength=0.5)

# Step 2: Apply CLAHE for contrast enhancement
print("\n🔹 Enhancing contrast with CLAHE...")
image_paths = sorted(glob.glob(os.path.join(UPSCALE_OUTPUT_DIR, "*.*")))
for img_path in image_paths:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        enhanced_image = adaptive_clahe(image)
        output_filename = os.path.basename(img_path).replace("_upscaled", "_clahe")
        cv2.imwrite(os.path.join(CLAHE_OUTPUT_DIR, output_filename), enhanced_image)

# Step 3: Apply Unsharp Masking for final sharpening
print("\n🔹 Applying adaptive sharpening...")
clahe_paths = sorted(glob.glob(os.path.join(CLAHE_OUTPUT_DIR, "*.*")))
for img_path in clahe_paths:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        sharpened_image = adaptive_sharpen(image)
        output_filename = os.path.basename(img_path).replace("_clahe", "_sharpened")
        cv2.imwrite(os.path.join(SHARPEN_OUTPUT_DIR, output_filename), sharpened_image)

print("\n✅ Processing complete! Sharpened images saved in:", SHARPEN_OUTPUT_DIR)
