"""
Advanced example demonstrating batch processing with the NSFW Image Detector package.
"""

import os
from PIL import Image
import torch
from nsfw_image_detector import NSFWDetector, NSFWLevel

def process_directory(directory_path, output_file, threshold_level=NSFWLevel.MEDIUM, threshold=0.5):
    """
    Process all images in a directory and save results to a file.
    
    Args:
        directory_path: Path to the directory containing images
        output_file: Path to save the results
        threshold_level: NSFW level to check against
        threshold: Probability threshold for classification
    """
    # Initialize the detector with bfloat16 for better performance
    detector = NSFWDetector(dtype=torch.bfloat16, device="cuda")
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    # Load all images
    images = []
    for img_file in image_files:
        try:
            img_path = os.path.join(directory_path, img_file)
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    # Process all images in a single batch
    probabilities = detector.predict_proba(images)
    is_nsfw = detector.is_nsfw(images, threshold_level, threshold)
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write(f"NSFW Detection Results (threshold_level={threshold_level}, threshold={threshold})\n")
        f.write("-" * 80 + "\n\n")
        
        for i, (img_file, probs, nsfw) in enumerate(zip(image_files, probabilities, is_nsfw)):
            f.write(f"Image: {img_file}\n")
            f.write(f"Is NSFW: {nsfw}\n")
            f.write("Probability scores:\n")
            for level, score in probs.items():
                f.write(f"  {level}: {score:.4f}\n")
            f.write("\n" + "-" * 40 + "\n\n")
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    directory = "path/to/your/images"
    output = "nsfw_detection_results.txt"
    
    # Process with medium threshold
    process_directory(directory, output, NSFWLevel.MEDIUM, 0.5)
    
    # Process with low threshold
    process_directory(directory, "nsfw_detection_results_low.txt", NSFWLevel.LOW, 0.5) 