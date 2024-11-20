import os
import sys
import tempfile
from io import BytesIO
from PIL import Image
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from helper_fns import gan_utils, cv_utils
from typing import Union

OUTPUT_DIR = "/Users/akhilbabu/Documents/work/Signature-Verification/output"


def main(document_image_path: str) -> None:
    """
    Main function to detect and clean a signature from a document image.

    Args:
        document_image_path (str): Path to the document image.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Detect signatures in the document
    detected_images = detect_signature(document_image_path)
    if not detected_images:
        print("Signature detection failed.")
        return

    # Get the first detected signature.
    signature_cropped_image_buffer = detected_images[0]

    # Step 2: Clean the detected signature
    print("About to call clean_signature")
    cleaned_image = clean_signature(signature_cropped_image_buffer)
    if not cleaned_image:
        print("Signature cleaning failed.")
        return

    # Save the final cleaned image to the output directory
    output_cleaned_path = os.path.join(OUTPUT_DIR, f"cleaned_signature.jpg")
    with open(output_cleaned_path, "wb") as f:
        f.write(cleaned_image.read())
    print(f"Cleaned signature saved to {output_cleaned_path}")


def detect_signature(document_image_path: str) -> list[BytesIO]:
    """
    Detects signatures in the document image using YOLO,
    applies make_square to each detected signature, and returns in-memory images.

    Args:
        document_image_path (str): Path to the document image.

    Returns:
        list[BytesIO]: List of BytesIO objects containing the processed signature images.
    """
    detected_images = detect.detect(document_image_path)
    processed_images = []

    if detected_images:
        for img_buffer in detected_images:
            # Load the image from the buffer and process it
            img_buffer.seek(0)
            detected_image = Image.open(img_buffer).convert("RGB")
            squared_image = gan_utils.make_square(detected_image)

            # Save the squared image back to a new BytesIO object
            new_img_buffer = BytesIO()
            squared_image.save(new_img_buffer, format="JPEG")
            new_img_buffer.seek(0)
            processed_images.append(new_img_buffer)

    return processed_images


def clean_signature(image_buffer: BytesIO) -> Union[BytesIO, None]:
    """
    Cleans the signature image using GAN processing and applies high-contrast cleaning
    as the final step.

    Args:
        image_buffer (BytesIO): BytesIO object of the detected signature image.

    Returns:
        Union[BytesIO, None]: BytesIO object of the cleaned and processed image or None on failure.
    """
    # Save the original sys.argv
    original_argv = sys.argv

    # Convert the input BytesIO object to an image and save it for GAN processing
    input_image = Image.open(image_buffer)
    temp_dir = tempfile.TemporaryDirectory()
    gan_input_dir = os.path.join(temp_dir.name, "input")
    os.makedirs(gan_input_dir, exist_ok=True)

    input_image_path = os.path.join(gan_input_dir, "input.jpg")
    input_image.save(input_image_path)

    # Temporarily set sys.argv to only include the script name
    sys.argv = [sys.argv[0]]

    try:
        # Run the GAN cleaning process
        cleaned_images = test.clean(input_image_path)
    finally:
        # Restore the original sys.argv
        sys.argv = original_argv

    if not cleaned_images or len(cleaned_images) < 2:
        print("GAN cleaning failed or returned incomplete results.")
        temp_dir.cleanup()
        return None

    # Get the second image as the first one is the input itself.
    cleaned_low_contrast_buffer = cleaned_images[1]
    cleaned_low_contrast_buffer.seek(0)
    cleaned_image = Image.open(cleaned_low_contrast_buffer)

    # Apply high-contrast cleaning
    high_contrast_image = cv_utils.high_contrast_clean(cleaned_image)

    # Save the high-contrast image back to a BytesIO object
    final_img_buffer = BytesIO()
    high_contrast_image.save(final_img_buffer, format="JPEG")
    final_img_buffer.seek(0)

    temp_dir.cleanup()
    return final_img_buffer


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python non_uifiles.py <document_image_path>"
        )
        sys.exit(1)

    document_image_path = sys.argv[1]
    main(document_image_path)
    print("Processing completed.")
