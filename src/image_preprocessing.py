import pandas as pd
import numpy as np
import cv2


def clahe_10_rescale(img):
    """
    Because ImageDataGenerator accepts preprocessing function that only gets the input image as a parameter, I created
    this function to apply CLAHE with the second clipLimit used in my experiments
    """
    eq = clahe(img, clip=10)
    return rescale(eq)


def clahe_rescale(img, clip=2, tile=(8, 8)):
    """
    Preprocessing pipeline for applying our CLAHE function and then rescaling the image.
    """
    eq = clahe(img, clip, tile)
    return rescale(eq)


def clahe(img, clip=2, tile=(8, 8)):
    """
    Applies CLAHE to the luminance channel of a 3-channel image in LAB color

    Parameters
    ----------
    img: np.ndarray
        Image to be processed
    clip: float
        Value of clipLimit parameter for opencv CLAHE function
    tile: tuple
        Size of grid tiles used in opencv CLAHE function

    Returns
    -------
    3-channel image preprocessed with CLAHE
    """
    img = img.astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def eq_hist_rescale(img):
    """
    Preprocessing pipeline for applying our histogram equalization function and then rescaling the image.
    """
    eq = eq_hist(img)
    return rescale(eq)


def eq_hist(img):
    """
    Applies histogram equalization to the luminance channel of a 3-channel image in LAB color

    Parameters
    ----------
    img: np.ndarray
        Image to be processed

    Returns
    -------
    3-channel image preprocessed with histogram equalization
    """
    img = img.astype('uint8')
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def rescale(img):
    """
    Scale pixel values to [0, 1] range
    """
    return img * 1. / 255


def bins_dominant_color(img):
    """
    For given 3-channel image, compute the most dominant color using bincount and return it as a single integer
    representing grayscale pixel intensity value.

    Code was taken from: https://stackoverflow.com/a/50900143
    """
    img2D = img.reshape(-1, img.shape[-1])
    col_range = (256, 256, 256)
    img1D = np.ravel_multi_index(img2D.T, col_range)
    return np.unravel_index(np.bincount(img1D).argmax(), col_range)[0]


def image_crop(img, box_ratio=0.05, contour_ratio=0.3, save_steps=False):
    """
    Custom preprocessing method consisting of multiple steps. Firstly, the dominant color in the form of a grayscale
    pixel intensity is determined. If the image is mostly white, it is inverted The image is converted to grayscale,
    then CLAHE and gaussian blur are applied. Binary threshold with Otsu's method to determine the threshold value is
    used to create a mask, then a bounding rectangle area is fit to it. If the area is bigger than given threshold,
    the image is cropped accordingly and another binary threshold with Otsu's method is used. Contours are found using
    the second mask and the biggest one is transformed into a convex hull. If the hull area is bigger than given
    threshold a mask created from it is used to filter out pixels outside the hull area and image is cropped again.

    Parameters
    ----------
    img: np.ndarray
        Input image to be processed
    box_ratio: float
        Determines minimal portion of the image area the bounding rectangle must cover in order for the image to be cropped
    contour_ratio: float
        Determines minimal portion of the image area the convex hull must cover in order for the mask to be applied
    save_steps: bool
        Whether to return only the processed image or also images representing specific preprocessing steps, used
        for visualisation of the process.

    Returns
    -------
    Final preprocessed image or multiple images representing specific function steps.
    """
    orig_img = img.copy()
    # Determine dominant color using bins
    dominant_color = bins_dominant_color(img)
    # If image is white-ish, invert it
    if dominant_color > 127:
        img = 255 - img

    # Convert to grayscale and apply grayscale CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    eq = gray_clahe.apply(gray)

    # Blur image
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    # Apply binary threshold using Otsu's method
    thresh_box = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Get bounding rectangle for binary threshold
    x, y, w, h = cv2.boundingRect(thresh_box)

    box_img = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  # Image with drawn bounding rectangle, for visualisation
    cv2.rectangle(box_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cropped_img = blur.copy()  # Image to be cropped

    # If the bounding rectangle area is bigger than given threshold, crop image according to the rectangle
    if (w * h) > (img.shape[0] * img.shape[1] * box_ratio):
        cropped_img = cropped_img[y:y + h, x:x + w]

    # Apply binary threshold using Otsu's method
    thresh_contours = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find all contours and select the biggest one
    contours = cv2.findContours(thresh_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    max_cont = max(contours, key=cv2.contourArea)

    # Create convex hull from the biggest contour and draw it
    hull = cv2.convexHull(max_cont, False)
    hull_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(hull_img, [hull], 0, (0, 255, 0), 3)

    # Prepare mask for bitwise and, initially fully black
    mask = np.zeros((thresh_contours.shape[0], thresh_contours.shape[1]), np.uint8)

    # If convex hull area is bigger than given threshold, create mask from bitwise and
    if cv2.contourArea(hull) > (cropped_img.shape[0] * cropped_img.shape[1] * contour_ratio):
        cv2.drawContours(mask, [hull], 0, 255, -1)
    else:
        mask = 255 - mask  # We want to keep the whole image, mask should be white

    # Apply bitwise and to remove unwanted regions from image
    masked_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

    # Get bounding rectangle from hull mask and apply second cropping if possible
    x, y, w, h = cv2.boundingRect(mask)
    final_img = masked_img[y:y + h, x:x + w]

    if save_steps:
        return orig_img, blur, box_img, cropped_img, hull_img, final_img
    else:
        return final_img
