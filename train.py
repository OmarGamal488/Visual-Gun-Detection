import argparse
from gun_detector import GunDetector

def main():
    parser = argparse.ArgumentParser(description='Train the Gun Detection System')
    parser.add_argument('--train_dir', type=str, default='dataset/training/guns',
                      help='Directory containing training images')
    parser.add_argument('--k_clusters', type=int, default=5,
                      help='Number of clusters for K-means segmentation')
    
    args = parser.parse_args()
    
    detector = GunDetector()
    detector.k_clusters = args.k_clusters
    
    print(f"Training on images in {args.train_dir}")
    detector.train(args.train_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()