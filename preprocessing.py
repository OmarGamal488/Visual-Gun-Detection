import cv2

def resize_image(image, width=400, height=300):
    return cv2.resize(image, (width, height))

def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def preprocess_image(image, width=400, height=300, kernel_size=5):
    if image.shape[1] != width or image.shape[0] != height:
        image = resize_image(image, width, height)
    
    filtered_image = apply_median_filter(image, kernel_size)
    
    return filtered_image