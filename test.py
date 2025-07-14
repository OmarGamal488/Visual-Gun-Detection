
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
from gun_detector import GunDetector
from descriptor_manager import DescriptorManager
from utils import load_image, save_visualization

def main():
    parser = argparse.ArgumentParser(description='Test the Gun Detection System')
    parser.add_argument('--test_dir', type=str, default='dataset/testing',
                      help='Directory containing test images')
    parser.add_argument('--ground_truth', type=str, default='dataset/ground_truth.csv',
                      help='Path to ground truth file')
    parser.add_argument('--method', type=str, default='freak',  
                      help='Feature method to use (sift, orb, freak)')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to a single image for testing (optional)')
    
    args = parser.parse_args()
    
    detector = GunDetector()    
    desc_manager = DescriptorManager()
    
    descriptors = desc_manager.load_method(args.method)    
    detector.descriptors[args.method] = descriptors
    detector.set_method(args.method)
    
    if args.image:
        print(f"Testing on single image: {args.image}")
        image = load_image(args.image)
        
        is_gun, _, vis = detector.detect(image, method=args.method, visualize_steps=True)
        
        result_text = "GUN DETECTED" if is_gun else "NO GUN DETECTED"
        print(f"Detection result ({args.method.upper()}): {result_text}")
        
        if vis:
            save_visualization(vis, f"{args.method}_detection_result.png", 'output/results')
            plt.show()
    
    elif args.test_dir and os.path.exists(args.test_dir):
        if not os.path.exists(args.ground_truth):
            print(f"Ground truth file {args.ground_truth} not found.")
            print("Creating a sample ground truth file...")
            
            # create_ground_truth(args.test_dir, args.ground_truth)
        
        print(f"Evaluating on test directory: {args.test_dir}")
        print(f"Using feature method: {args.method.upper()}")
        
        ground_truth = {}
        with open(args.ground_truth, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        is_gun = parts[1].strip().lower() == 'gun'
                        ground_truth[filename] = is_gun
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(args.test_dir) 
                              if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No images found in {args.test_dir}")
            return
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        results_dir = os.path.join('output', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for filename in tqdm(image_files, desc=f"Evaluating with {args.method.upper()}"):
            if filename not in ground_truth:
                print(f"Warning: No ground truth for {filename}, skipping")
                continue
            
            image_path = os.path.join(args.test_dir, filename)
            image = load_image(image_path)
            
            is_gun_detected, _, _ = detector.detect(image, method=args.method, visualize_steps=False)            
            is_gun_expected = ground_truth[filename]
            
            if is_gun_detected and is_gun_expected:
                true_positives += 1
            elif is_gun_detected and not is_gun_expected:
                false_positives += 1
            elif not is_gun_detected and not is_gun_expected:
                true_negatives += 1
            else:
                false_negatives += 1
            
            vis_image = image.copy()
            result_text = "GUN DETECTED" if is_gun_detected else "NO GUN DETECTED"
            expected_text = "EXPECTED: GUN" if is_gun_expected else "EXPECTED: NO GUN"
            
            if is_gun_detected == is_gun_expected:
                color = (0, 255, 0) 
            else:
                color = (0, 0, 255)
            
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 60), (0, 0, 0), -1)
            
            alpha = 0.6
            vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
            
            cv2.putText(vis_image, f"{result_text} | {expected_text}", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            output_path = os.path.join(results_dir, f"result_{filename}")
            cv2.imwrite(output_path, vis_image)
        
        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n===== Evaluation Results =====")
        print(f"Feature Method: {args.method.upper()}")
        print(f"Total images: {total}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        plt.figure(figsize=(8, 6))
        cm = np.array([
            [true_negatives, false_positives],
            [false_negatives, true_positives]
        ])
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {args.method.upper()}')
        plt.colorbar()
        
        classes = ['Non-Gun', 'Gun']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(os.path.join(results_dir, f"confusion_matrix_{args.method}.png"))
        plt.close()
        
        with open(os.path.join(results_dir, f"evaluation_results_{args.method}.txt"), 'w') as f:
            f.write(f"===== Evaluation Results - {args.method.upper()} =====\n")
            f.write(f"Total images: {total}\n")
            f.write(f"True Positives: {true_positives}\n")
            f.write(f"False Positives: {false_positives}\n")
            f.write(f"True Negatives: {true_negatives}\n")
            f.write(f"False Negatives: {false_negatives}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")
        
        print(f"\nResults saved to {results_dir}")
    else:
        print(f"Test directory {args.test_dir} not found.")
        
# def create_ground_truth(test_dir, output_file):
#     image_extensions = ['.jpg', '.jpeg', '.png']
#     image_files = []
    
#     for ext in image_extensions:
#         image_files.extend([f for f in os.listdir(test_dir) 
#                           if f.lower().endswith(ext)])
    
#     with open(output_file, 'w') as f:
#         for filename in image_files:
#             is_gun = 'armas' in filename.lower() or 'gun' in filename.lower()
#             label = 'gun' if is_gun else 'non_gun'
#             f.write(f"{filename},{label}\n")
    
#     print(f"Created sample ground truth file with {len(image_files)} entries at {output_file}")
#     print("Please review and edit the file to ensure correct labeling.")

if __name__ == "__main__":
    main()
