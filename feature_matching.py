import cv2
import numpy as np
import matplotlib.pyplot as plt

def ssd_distance(desc1, desc2):
    return np.sum((desc1 - desc2) ** 2)

def match_features(descriptors, gun_descriptors_db, ratio_threshold=0.75, visualize=False):
    if descriptors is None or len(descriptors) == 0:
        return 0, 0, None
    
    matches = 0
    total = len(descriptors)
    matching_details = []
    
    for i, desc in enumerate(descriptors):
        best_distance = float('inf')
        second_best_distance = float('inf')
        best_match_idx = -1
        best_class_idx = -1
        
        for class_idx, gun_descriptors in enumerate(gun_descriptors_db):
            for j, ref_desc in enumerate(gun_descriptors):
                distance = ssd_distance(desc, ref_desc)
                
                if distance < best_distance:
                    second_best_distance = best_distance
                    best_distance = distance
                    best_match_idx = j
                    best_class_idx = class_idx
                elif distance < second_best_distance:
                    second_best_distance = distance
        
        if best_distance < float('inf') and second_best_distance < float('inf'):
            if second_best_distance < 0.0001:
                ratio = 1.0
            else:
                ratio = best_distance / (second_best_distance * 1.0)
            
            if ratio < ratio_threshold:
                matches += 1
                matching_details.append({
                    'descriptor_idx': i,
                    'best_match_idx': best_match_idx,
                    'best_class_idx': best_class_idx,
                    'ratio': ratio,
                    'distance': best_distance
                })
    
    match_ratio = matches / total if total > 0 else 0
    
    visualization = None
    if visualize:
        ratios = [match['ratio'] for match in matching_details]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if ratios:
            axes[0].hist(ratios, bins=20, range=(0, 1), color='blue', alpha=0.7)
            axes[0].set_title('Ratio Distribution')
            axes[0].set_xlabel('Ratio (lower is better)')
            axes[0].set_ylabel('Count')
            axes[0].axvline(x=ratio_threshold, color='red', linestyle='--', 
                           label=f'Threshold ({ratio_threshold})')
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'No matches found', ha='center', va='center')
            axes[0].set_title('Ratio Distribution (Empty)')
        
        axes[1].bar(['Matches', 'Non-Matches'], [matches, total - matches], color=['green', 'red'])
        axes[1].set_title(f'Match Summary: {matches}/{total} ({match_ratio:.2f})')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        visualization = fig
    
    return match_ratio, matching_details, visualization