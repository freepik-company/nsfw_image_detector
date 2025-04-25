"""
Simple example demonstrating how to use the NSFW Image Detector package.
"""

from PIL import Image
from nsfw_image_detector import NSFWDetector, NSFWLevel

# Initialize the detector
detector = NSFWDetector()

# Load an image
image = Image.open("path/to/your/image.jpg")

# Get probability scores for all categories
probabilities = detector.predict_proba(image)
print("Probability scores:", probabilities)

# Check if the image contains NSFW content at medium level or higher
is_nsfw = detector.is_nsfw(image, threshold_level=NSFWLevel.MEDIUM)
print(f"Is NSFW (medium or higher): {is_nsfw}")

# Check if the image contains NSFW content at low level or higher
is_nsfw_low = detector.is_nsfw(image, threshold_level=NSFWLevel.LOW)
print(f"Is NSFW (low or higher): {is_nsfw_low}")

# Process multiple images
images = [
    Image.open("path/to/image1.jpg"),
    Image.open("path/to/image2.jpg"),
    Image.open("path/to/image3.jpg"),
]

# Get probability scores for all images
all_probabilities = detector.predict_proba(images)
print("Probability scores for all images:", all_probabilities)

# Check if any of the images contain NSFW content
all_is_nsfw = detector.is_nsfw(images)
print(f"Are images NSFW (medium or higher): {all_is_nsfw}") 