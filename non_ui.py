import os
import sys
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
import shutil

# Import your custom modules
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify

def main(document_image_path, signature_image_path):
    # Step 1: Signature Detection using your custom YOLOv5 code
    detected_signature_image_path = detect_signature(document_image_path)
    if detected_signature_image_path is None:
        print("Signature detection failed.")
        return None

    # Step 2: Signature Cleaning using your GAN code
    cleaned_signature_image_path = clean_signature(detected_signature_image_path)
    if cleaned_signature_image_path is None:
        print("Signature cleaning failed.")
        return None

    # Step 3: Signature Verification using your VGG model
    similarity = verify_signature(cleaned_signature_image_path, signature_image_path)
    if similarity is None:
        print("Signature verification failed.")
        return None

    return similarity

def detect_signature(document_image_path):
    """
    Detects signature in the document image using your custom YOLOv5 code.

    Args:
        document_image_path (str): Path to the document image.

    Returns:
        str: Path to the cropped signature image, or None if no signature detected.
    """
    # Call your custom detect function
    detect.detect(document_image_path)

    # Assuming your detect function saves the cropped signatures in 'results/yolov5/exp/crops/DLSignature/'
    # Adjust the paths based on your actual output directories
    yolo_results_dir = 'results/yolov5'
    latest_exp_dir = max([os.path.join(yolo_results_dir, d) for d in os.listdir(yolo_results_dir) if d.startswith('exp')], key=os.path.getmtime)
    crops_dir = os.path.join(latest_exp_dir, 'crops', 'DLSignature')

    # Get the list of cropped signature images
    cropped_images = [os.path.join(crops_dir, f) for f in os.listdir(crops_dir) if f.endswith(('.jpg', '.png'))]

    if not cropped_images:
        print("No signature detected.")
        return None

    # For simplicity, select the first cropped image
    detected_signature_image_path = cropped_images[0]

    return detected_signature_image_path

# def clean_signature(signature_image_path):
#     """
#     Cleans the signature image using your GAN code.

#     Args:
#         signature_image_path (str): Path to the signature image to clean.

#     Returns:
#         str: Path to the cleaned signature image.
#     """
#     # Prepare the input directory for the GAN
#     gan_input_dir = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
#     os.makedirs(gan_input_dir, exist_ok=True)

#     # Clear any existing images in the GAN input directory
#     for f in os.listdir(gan_input_dir):
#         os.remove(os.path.join(gan_input_dir, f))

#     # Copy the signature image to the GAN input directory
#     shutil.copy(signature_image_path, gan_input_dir)

#     # Run the GAN cleaning process
#     test.clean()

#     # The cleaned images are saved in 'results/gan/gan_signdata_kaggle/test_latest/images/'
#     gan_output_dir = 'results/gan/gan_signdata_kaggle/test_latest/images/'

#     # Get the cleaned image
#     cleaned_images = [os.path.join(gan_output_dir, f) for f in os.listdir(gan_output_dir) if 'fake' in f]

#     if not cleaned_images:
#         print("No cleaned signature image found.")
#         return None

#     # For simplicity, select the first cleaned image
#     cleaned_signature_image_path = cleaned_images[0]

#     return cleaned_signature_image_path

import sys

def clean_signature(signature_image_path):
    """
    Cleans the signature image using your GAN code.

    Args:
        signature_image_path (str): Path to the signature image to clean.

    Returns:
        str: Path to the cleaned signature image.
    """
    # Prepare the input directory for the GAN
    gan_input_dir = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
    os.makedirs(gan_input_dir, exist_ok=True)

    # Clear any existing images in the GAN input directory
    for f in os.listdir(gan_input_dir):
        os.remove(os.path.join(gan_input_dir, f))

    # Copy the signature image to the GAN input directory
    shutil.copy(signature_image_path, gan_input_dir)

    # Save the original sys.argv
    original_argv = sys.argv
    # Temporarily set sys.argv to only include the script name
    sys.argv = [sys.argv[0]]

    # Run the GAN cleaning process
    test.clean()

    # Restore the original sys.argv
    sys.argv = original_argv

    # The cleaned images are saved in 'results/gan/gan_signdata_kaggle/test_latest/images/'
    gan_output_dir = 'results/gan/gan_signdata_kaggle/test_latest/images/'

    # Get the cleaned image
    cleaned_images = [os.path.join(gan_output_dir, f) for f in os.listdir(gan_output_dir) if 'fake' in f]

    if not cleaned_images:
        print("No cleaned signature image found.")
        return None

    # For simplicity, select the first cleaned image
    cleaned_signature_image_path = cleaned_images[0]

    return cleaned_signature_image_path
def verify_signature(cleaned_signature_image_path, signature_image_path):
    """
    Verifies the cleaned signature image against the provided signature image.

    Args:
        cleaned_signature_image_path (str): Path to the cleaned signature image.
        signature_image_path (str): Path to the signature image to compare with.

    Returns:
        float: Similarity value between 0 and 1.
    """
    # Prepare the directory for verification
    verification_dir = os.path.dirname(cleaned_signature_image_path)

    # Run the verification
    feature_set = vgg_verify.verify(signature_image_path, verification_dir)

    # Find the similarity score for the cleaned signature image
    for image_path, score in feature_set:
        if os.path.basename(image_path) == os.path.basename(cleaned_signature_image_path):
            return score

    print("Cleaned signature image not found in verification results.")
    return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python signature_verification.py <document_image_path> <signature_image_path>")
        sys.exit(1)
    document_image_path = sys.argv[1]
    signature_image_path = sys.argv[2]
    similarity = main(document_image_path, signature_image_path)
    if similarity is not None:
        print(f"Similarity: {similarity}")
