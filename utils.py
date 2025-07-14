import cv2
import matplotlib.pyplot as plt
import os

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")
    return image

def display_images(images, titles=None, figsize=(15, 10), rows=None, cols=None):
    num_images = len(images)
    
    if rows is None and cols is None:
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
    elif rows is None:
        rows = (num_images + cols - 1) // cols
    elif cols is None:
        cols = (num_images + rows - 1) // rows
    
    plt.figure(figsize=figsize)
    
    for i, image in enumerate(images):
        if i < num_images:
            plt.subplot(rows, cols, i + 1)            
            if len(image.shape) == 2 or image.shape[2] == 1:
                plt.imshow(image, cmap='gray')
            else:
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    plt.imshow(image_rgb)
                else:
                    plt.imshow(image)
            
            if titles is not None and i < len(titles):
                plt.title(titles[i])
            
            plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def save_visualization(fig, filename, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {filepath}")

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    """Draw keypoints on an image"""
    image_with_keypoints = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(image_with_keypoints, (x, y), 5, color, 1)
    return image_with_keypoints

def create_output_dirs():
    dirs = ['output', 'output/preprocessed', 'output/segmentation', 
            'output/morphology', 'output/features', 'output/results',
            'descriptors']
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")