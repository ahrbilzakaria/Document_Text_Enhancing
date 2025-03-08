import cv2
import numpy as np

def adaptive_clahe(image, clip_limit=2.0, tile_size=8):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate image brightness (mean pixel intensity)
    brightness = np.mean(image)

    # Adjust clip limit based on brightness
    if brightness < 50:  # Dark image
        clip_limit = min(clip_limit + 1.0, 4.0)
    elif brightness > 200:  # Bright image
        clip_limit = max(clip_limit - 1.0, 1.0)

    # Adjust tile size based on image resolution
    height, width = image.shape
    if max(height, width) > 2000:  # High-resolution image
        tile_size = 16
    else:
        tile_size = 8

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    clahe_image = clahe.apply(image)

    return clahe_image

# Load the image
image = cv2.imread('./output/IMG_20250306_151247_upscaled.jpg')

# Apply adaptive CLAHE
clahe_image = adaptive_clahe(image)

# Save or display the result
cv2.imwrite('adaptive_clahe_output.jpg', clahe_image)
cv2.imshow('Adaptive CLAHE Image', clahe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()