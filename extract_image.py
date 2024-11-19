import numpy as np
import cv2
import sys
from helper_fns import cv_utils



def extract_signature_gray(image_path, output_image_path):
    '''
    Extracts and cleans the signature from a grayscale image at image_path,
    upscales it if necessary, and saves the result to output_image_path.
    '''
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded properly
    if image is None:
        print(f"Error loading image {image_path}")
        return False

    # Upscale the image if its resolution is below the threshold
    image = cv_utils.upscale_image(image)

    cv2.imshow('image_upscaled', image)

    # Remove black padding from the image
    image = cv_utils.remove_black_padding(image)

    # Apply slight gamma correction for subtle contrast enhancement
    gamma = 0.7  # Values >1 increase brightness, <1 decrease brightness
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, look_up_table)

    # Apply Gaussian Blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply Otsu's thresholding to binarize the image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply a mild morphological transformation to connect small gaps
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Resize the binary image to 512x512 with black padding
    binary = cv_utils.make_square(binary)

    # Save the binary result directly
    cv2.imwrite(output_image_path, binary)

    # Display the results (optional)
    cv2.imshow('Binary Image', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Processed image saved at {output_image_path}")
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_signature_gray.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    # Call the extract_signature_gray function
    success = extract_signature_gray(input_image_path, output_image_path)
    if success:
        print(f"Processed image saved successfully.")
    else:
        print("Failed to process the image.")

if __name__ == "__main__":
    main()
