import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_harris_corners(image, mask, blockSize=2, ksize=3, k=0.04, max_corners=100, visualize=False):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    if len(masked_image.shape) == 3:
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = masked_image
    
    corner_response = cv2.cornerHarris(gray, blockSize, ksize, k)
    
    corner_response = cv2.dilate(corner_response, None)
    
    threshold = 0.015 * corner_response.max()
    corner_mask = corner_response > threshold
    
    keypoints = []
    y_coords, x_coords = np.where(corner_mask)
    for x, y in zip(x_coords, y_coords):
        response = corner_response[y, x]
        keypoints.append(cv2.KeyPoint(float(x), float(y), 7, -1, float(response), 0, -1))
    
    if keypoints:
        keypoints.sort(key=lambda x: x.response)
        keypoints.reverse()
        keypoints = keypoints[:max_corners]
    
    visualization = None
    if visualize and keypoints:
        corner_image = image.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(corner_image, (x, y), 3, (0, 0, 255), -1)
        
        keypoint_image = cv2.drawKeypoints(
            image, keypoints, None, 
            color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Masked Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Harris Corners')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Keypoints ({len(keypoints)})')
        axes[2].axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return keypoints, corner_response, visualization

def extract_freak_descriptors(image, keypoints, visualize=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    freak = cv2.xfeatures2d.FREAK_create(
        orientationNormalized=True,
        scaleNormalized=True,
        patternScale=0.7,
        nOctaves=4
    )
    
    keypoints, descriptors = freak.compute(gray, keypoints)
    
    visualization = None
    if visualize and descriptors is not None and len(descriptors) > 0:
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None, 
            color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f'FREAK Features ({len(descriptors)} descriptors)')
        ax.axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return keypoints, descriptors, visualization

def extract_sift_features(image, mask, visualize=False):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    if len(masked_image.shape) == 3:
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = masked_image
    
    sift = cv2.SIFT_create()
    
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    
    visualization = None
    if visualize and keypoints and descriptors is not None:
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None, 
            color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f'SIFT Features ({len(keypoints)} keypoints)')
        ax.axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return keypoints, descriptors, visualization

def extract_orb_features(image, mask, visualize=False):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    if len(masked_image.shape) == 3:
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = masked_image
    
    orb = cv2.ORB_create()
    
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    
    visualization = None
    if visualize and keypoints and descriptors is not None:
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None, 
            color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f'ORB Features ({len(keypoints)} keypoints)')
        ax.axis('off')
        
        plt.tight_layout()
        visualization = fig
    
    return keypoints, descriptors, visualization

def extract_features_by_method(image, mask, method='freak', visualize=False):
    if method.lower() == 'sift':
        return extract_sift_features(image, mask, visualize)
    elif method.lower() == 'orb':
        return extract_orb_features(image, mask, visualize)
    elif method.lower() == 'freak':
        keypoints, _, corner_viz = detect_harris_corners(
            image, 
            mask, 
            max_corners=200,
            visualize=visualize
        )
        keypoints_desc, descriptors, desc_viz = extract_freak_descriptors(image, keypoints, visualize)
        
        visualization = desc_viz if desc_viz is not None else corner_viz
        
        return keypoints_desc, descriptors, visualization
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

def compare_feature_detectors(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    if len(masked_image.shape) == 3:
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = masked_image
    
    sift_keypoints, sift_descriptors = extract_sift_features(image, mask)[:2]
    
    orb_keypoints, orb_descriptors = extract_orb_features(image, mask)[:2]
    
    harris_keypoints, _, _ = detect_harris_corners(image, mask, max_corners=100)
    freak = cv2.xfeatures2d.FREAK_create()
    harris_keypoints, freak_descriptors = freak.compute(gray, harris_keypoints) if harris_keypoints else ([], None)
    
    sift_img = cv2.drawKeypoints(
        image, sift_keypoints if sift_keypoints else [], None, 
        color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    orb_img = cv2.drawKeypoints(
        image, orb_keypoints if orb_keypoints else [], None, 
        color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    harris_img = cv2.drawKeypoints(
        image, harris_keypoints if harris_keypoints else [], None, 
        color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'SIFT ({len(sift_keypoints) if sift_keypoints else 0} keypoints)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'ORB ({len(orb_keypoints) if orb_keypoints else 0} keypoints)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Harris+FREAK ({len(harris_keypoints) if harris_keypoints else 0} keypoints)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    results = {
        'sift': {
            'keypoints': sift_keypoints,
            'descriptors': sift_descriptors,
            'count': len(sift_keypoints) if sift_keypoints else 0
        },
        'orb': {
            'keypoints': orb_keypoints,
            'descriptors': orb_descriptors,
            'count': len(orb_keypoints) if orb_keypoints else 0
        },
        'freak': {
            'keypoints': harris_keypoints,
            'descriptors': freak_descriptors,
            'count': len(harris_keypoints) if harris_keypoints else 0
        }
    }
    
    return results, fig

def save_method_descriptors(all_descriptors, method, output_dir='descriptors'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{method}_descriptors.npy")
    
    np.save(output_path, np.array(all_descriptors, dtype=object))
    print(f"Saved {len(all_descriptors)} {method.upper()} descriptor sets to {output_path}")
    return output_path
