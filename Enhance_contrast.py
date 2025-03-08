import cv2
import numpy as np

def adaptive_clahe(image, clip_limit=2.0, tile_size=8):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(image)

    # Dynamically adjust CLAHE parameters based on brightness
    if brightness < 50:
        clip_limit = min(clip_limit + 1.0, 4.0)  # Increase contrast for dark images
    elif brightness > 200:
        clip_limit = max(clip_limit - 1.0, 1.0)  # Reduce contrast for bright images

    height, width = image.shape
    tile_size = 16 if max(height, width) > 2000 else 8  # Adjust tile size for high-res images

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)
