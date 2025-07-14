import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_morphology(mask, close_kernel_size=7, open_kernel_size=5, min_area=1000, visualize=False):
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    visualization = None
    if visualize:
        stages = [
            ('Original Mask', mask),
            ('Closing', closed),
            ('Opening', opened),
            ('Filtered Contours', filtered_mask)
        ]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for i, (title, img) in enumerate(stages):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return filtered_mask, visualization

def extract_blobs(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 1000
    blobs = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            blob_mask = np.zeros_like(mask)
            cv2.drawContours(blob_mask, [contour], 0, 255, -1)
            blobs.append(blob_mask)
    
    return blobs
