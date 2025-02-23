import cv2
import numpy as np

def postprocess_mask(mask, kernel_size=3):
    # Convert mask to binary image (0s and 255s)
    mask_binary = (mask > 0.5).astype(np.uint8) * 255

    # Convert to OpenCV's format (0s and 1s)
    mask_binary = mask_binary.astype(np.uint8)

    # Define a structuring element (kernel) for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply morphological closing to fill small holes in the mask
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # Apply morphological opening to remove small noise
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    return mask_opened

# Example usage:
# Assuming 'prediction' contains the predicted binary mask
# Apply post-processing to the predicted mask
postprocessed_mask = postprocess_mask(prediction)

# Now 'postprocessed_mask' contains the refined binary mask after morphology operations
