"""
Image preprocessing utilities for OCR improvement
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(image: np.ndarray, 
                    deskew: bool = True,
                    binarize: bool = True,
                    noise_removal: bool = True) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy
    
    Args:
        image: Input image as numpy array
        deskew: Whether to correct skew
        binarize: Whether to convert to binary
        noise_removal: Whether to remove noise
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    # Convert to grayscale if needed
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Deskew
    if deskew:
        processed = correct_skew(processed)
    
    # Binarization
    if binarize:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    if noise_removal:
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed = cv2.medianBlur(processed, 3)
    
    return processed


def correct_skew(image: np.ndarray) -> np.ndarray:
    """
    Correct image skew using Hough transform
    
    Args:
        image: Input grayscale image
        
    Returns:
        Deskewed image
    """
    # Convert to binary
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(binary, 1, np.pi / 180, 100)
    
    if lines is None or len(lines) == 0:
        return image
    
    # Calculate average angle
    angles = []
    for rho, theta in lines[:10]:  # Use first 10 lines
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 45:  # Only consider reasonable angles
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Rotate image to correct skew
    if abs(median_angle) > 0.5:  # Only rotate if significant skew
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, 
                              (image.shape[1], image.shape[0]),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    
    return image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV format to PIL Image"""
    if len(cv2_image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2_image)


