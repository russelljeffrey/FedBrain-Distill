import numpy as np
import cv2

def min_max_normalize(image: np.ndarray):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)

# CLAHE Algorithm
def selective_clahe(image: np.ndarray, mask: np.ndarray, clip_limit: int =50, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    tumor_image = min_max_normalize(image)
    enhanced_image = tumor_image.copy()

    tumor_region = tumor_image * mask
    tumor_region_clahe = clahe.apply(tumor_region)

    enhanced_image[mask == 1] = tumor_region_clahe[mask == 1]

    return enhanced_image

def enhance_tumor_images(tumor_images: np.ndarray, mask_images: np.ndarray):
    """
    This function enhances the tumor region using CLAHE algorithm

    Args:
        tumor_images (np.array): numpy array of tumor images
        mask_images (np.array): numpy array of tumor regions

    Returns:
        tumor images with tumor region enhanced
    """
    enhanced_images = []
    for i in range(len(tumor_images)):
        enhanced_image = selective_clahe(tumor_images[i], mask_images[i])
        enhanced_images.append(enhanced_image)
    enhanced_images = np.stack(enhanced_images)
    return enhanced_images

def color_channel_modification(tumor_images: np.ndarray):
    modified_tumor_images = []
    for i in range(len(tumor_images)):
        x = cv2.cvtColor(tumor_images[i], cv2.COLOR_GRAY2RGB)
        modified_tumor_images.append(x)
    return np.stack(modified_tumor_images)