import cv2
import numpy as np

def adaptive_sharpen(image, base_sigma=1.0, base_strength=1.5):
    """
    Applies unsharp masking with parameters dynamically adjusted to image properties.
    
    Args:
        image: Grayscale image (CLAHE-enhanced, 8-bit).
        base_sigma: Default blur strength (overridden adaptively).
        base_strength: Default sharpening strength (overridden adaptively).
    
    Returns:
        Sharpened grayscale image.
    """

    # 1. Adjust Sigma Based on Resolution
    height, width = image.shape[:2]
    
    # High-resolution: Upscaled images (e.g., 4x from Real-ESRGAN)
    if max(height, width) > 2000:  # >2K pixels on either side
        sigma = max(base_sigma, 2.0)  # Broaden blur to target text edges, not noise
    else:
        sigma = min(base_sigma, 1.0)  # Target finer details
    

    # 2. Adjust Strength Based on Contrast
    # Measure local contrast (std of pixel intensities)
    contrast = np.std(image)
    
    # Low-contrast text (e.g., faded documents)
    if contrast < 40:
        strength = min(base_strength + 0.8, 2.5)  # Aggressive sharpening
    # High-contrast text (e.g., clean scans)
    elif contrast > 70:
        strength = max(base_strength - 0.5, 0.8)  # Conservative sharpening
    # Moderate contrast (default)
    else:
        strength = base_strength
    

    # 3. Apply Unsharp Masking
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

