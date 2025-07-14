import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_segmentation(image, k=5, visualize=False):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    masks = []
    for i in range(k):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[labels.reshape(image.shape[:2]) == i] = 255
        masks.append(mask)
    
    visualization = None
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Segmented Image')
        axes[1].axis('off')
        
        for i in range(min(k, 4)):
            cluster_viz = cv2.bitwise_and(image, image, mask=masks[i])
            axes[i+2].imshow(cv2.cvtColor(cluster_viz, cv2.COLOR_BGR2RGB))
            avg_color = centers[i].tolist()
            axes[i+2].set_title(f'Cluster {i+1}: RGB={avg_color}')
            axes[i+2].axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return masks, segmented_image, visualization, centers

def select_gun_clusters(image, masks, centers, threshold=120):
    selected_masks = []
    
    avg_colors = [np.mean(center) for center in centers]
    
    for i, avg_color in enumerate(avg_colors):
        if avg_color < threshold:
            selected_masks.append(masks[i])
    
    if not selected_masks:
        darkest_idx = np.argmin(avg_colors)
        selected_masks.append(masks[darkest_idx])
    
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in selected_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask, selected_masks
