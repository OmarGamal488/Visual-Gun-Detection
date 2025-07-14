import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing import preprocess_image
from segmentation import kmeans_segmentation, select_gun_clusters
from morphology import apply_morphology, extract_blobs
from feature_extraction import extract_features_by_method, compare_feature_detectors, save_method_descriptors
from feature_matching import match_features
from utils import load_image, display_images, save_visualization, create_output_dirs

class GunDetector:
    def __init__(self):
        """Initialize the gun detector"""
        self.descriptors = {
            'sift': [],
            'orb': [],
            'freak': []
        }
        self.current_method = 'freak'
        self.k_clusters = 5
        self.match_threshold = 0.5
        self.visualize_training = True
        self.visualize_detection = True
    
    def train(self, training_dir):
        create_output_dirs()
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([os.path.join(training_dir, f) for f in os.listdir(training_dir) 
                            if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No images found in {training_dir}")
            return
        
        print(f"Found {len(image_files)} images for training")
        
        sift_descriptors = []
        orb_descriptors = []
        freak_descriptors = []
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing training images")):
            image = load_image(image_path)
            preprocessed = preprocess_image(image)
            
            filename = os.path.basename(image_path)
            
            if self.visualize_training:
                images = [image, preprocessed]
                titles = ['Original', 'Preprocessed']
                fig = display_images(images, titles, figsize=(10, 5))
                save_visualization(fig, f"preprocess_{filename}.png", 'output/preprocessed')
                plt.close(fig)
            
            masks, segmented, seg_viz, centers = kmeans_segmentation(preprocessed, self.k_clusters, 
                                                        visualize=self.visualize_training)
            
            if self.visualize_training and seg_viz:
                save_visualization(seg_viz, f"segmentation_{filename}.png", 'output/segmentation')
                plt.close(seg_viz)
            
            gun_mask, _ = select_gun_clusters(preprocessed, masks, centers)
            
            processed_mask, morph_viz = apply_morphology(gun_mask, visualize=self.visualize_training)
            
            if self.visualize_training and morph_viz:
                save_visualization(morph_viz, f"morphology_{filename}.png", 'output/morphology')
                plt.close(morph_viz)
            
            blobs = extract_blobs(processed_mask)
            
            for blob_idx, blob in enumerate(blobs):
                _, sift_desc, sift_viz = extract_features_by_method(
                    preprocessed, blob, method='sift', visualize=self.visualize_training
                )
                
                if self.visualize_training and sift_viz and sift_desc is not None:
                    save_visualization(sift_viz, f"sift_{filename}_blob{blob_idx}.png", 'output/features')
                    plt.close(sift_viz)
                
                _, orb_desc, orb_viz = extract_features_by_method(
                    preprocessed, blob, method='orb', visualize=self.visualize_training
                )
                
                if self.visualize_training and orb_viz and orb_desc is not None:
                    save_visualization(orb_viz, f"orb_{filename}_blob{blob_idx}.png", 'output/features')
                    plt.close(orb_viz)
                
                _, freak_desc, freak_viz = extract_features_by_method(
                    preprocessed, blob, method='freak', visualize=self.visualize_training
                )
                
                if self.visualize_training and freak_viz and freak_desc is not None:
                    save_visualization(freak_viz, f"freak_{filename}_blob{blob_idx}.png", 'output/features')
                    plt.close(freak_viz)
                
                if sift_desc is not None and len(sift_desc) > 0:
                    sift_descriptors.append(sift_desc)
                    print(f"Extracted {len(sift_desc)} SIFT descriptors from {filename} blob {blob_idx}")
                
                if orb_desc is not None and len(orb_desc) > 0:
                    orb_descriptors.append(orb_desc)
                    print(f"Extracted {len(orb_desc)} ORB descriptors from {filename} blob {blob_idx}")
                
                if freak_desc is not None and len(freak_desc) > 0:
                    freak_descriptors.append(freak_desc)
                    print(f"Extracted {len(freak_desc)} FREAK descriptors from {filename} blob {blob_idx}")
            
            if i % 5 == 0:
                results, comp_viz = compare_feature_detectors(preprocessed, processed_mask)
                if comp_viz:
                    save_visualization(comp_viz, f"comparison_{filename}.png", 'output/features')
                    plt.close(comp_viz)
        
        save_method_descriptors(sift_descriptors, 'sift')
        save_method_descriptors(orb_descriptors, 'orb')
        save_method_descriptors(freak_descriptors, 'freak')
        
        self.descriptors = {
            'sift': sift_descriptors,
            'orb': orb_descriptors,
            'freak': freak_descriptors
        }
        
        print("Training complete for all feature methods!")
        
        return self.descriptors
    
    def set_method(self, method):
        if method.lower() in ['sift', 'orb', 'freak']:
            self.current_method = method.lower()
            print(f"Current feature method set to: {self.current_method.upper()}")
        else:
            print(f"Unknown method: {method}. Using {self.current_method.upper()} instead.")
    
    def detect(self, image, method=None, visualize_steps=False):
        if method:
            self.set_method(method)
        
        if not self.descriptors[self.current_method]:
            print(f"No {self.current_method.upper()} descriptors loaded. Please load descriptors first.")
            return False, [], None
        
        preprocessed = preprocess_image(image)
        
        masks, segmented, seg_viz, centers = kmeans_segmentation(preprocessed, self.k_clusters, 
                                                      visualize=visualize_steps)
        
        gun_mask, selected_masks = select_gun_clusters(preprocessed, masks, centers)
        
        processed_mask, morph_viz = apply_morphology(gun_mask, visualize=visualize_steps)
        
        blobs = extract_blobs(processed_mask)
        
        blob_results = []
        
        for blob_idx, blob in enumerate(blobs):
            keypoints, descriptors, feature_viz = extract_features_by_method(
                preprocessed, blob, method=self.current_method, visualize=visualize_steps
            )
            
            if keypoints and descriptors is not None and len(descriptors) > 0:
                match_ratio, match_details, match_viz = match_features(
                    descriptors, self.descriptors[self.current_method], visualize=visualize_steps
                )
                
                is_gun = match_ratio >= self.match_threshold
                
                blob_results.append({
                    'blob_idx': blob_idx,
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'match_ratio': match_ratio,
                    'is_gun': is_gun,
                    'blob_mask': blob
                })
        
        is_gun_detected = any(result['is_gun'] for result in blob_results) if blob_results else False
        
        if visualize_steps or self.visualize_detection:
            vis_image = image.copy()
            
            result_text = "GUN DETECTED" if is_gun_detected else "NO GUN DETECTED"
            color = (0, 0, 255) if is_gun_detected else (0, 255, 0)
            
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 60), (0, 0, 0), -1)
            
            alpha = 0.6
            vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
            
            cv2.putText(vis_image, f"{self.current_method.upper()}: {result_text}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            stages = []
            titles = []
            
            stages.extend([cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                         cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)])
            titles.extend(['Original', 'Preprocessed'])
            
            gun_mask_viz = cv2.bitwise_and(preprocessed, preprocessed, mask=gun_mask)
            stages.append(cv2.cvtColor(gun_mask_viz, cv2.COLOR_BGR2RGB))
            titles.append('Gun Mask')
            
            morph_viz_img = cv2.bitwise_and(preprocessed, preprocessed, mask=processed_mask)
            stages.append(cv2.cvtColor(morph_viz_img, cv2.COLOR_BGR2RGB))
            titles.append('Morphology')
            
            stages.append(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            titles.append('Detection Result')
            
            fig = display_images(stages, titles, figsize=(15, 8))
            detection_visualization = fig
        else:
            detection_visualization = None
        
        return is_gun_detected, blob_results, detection_visualization