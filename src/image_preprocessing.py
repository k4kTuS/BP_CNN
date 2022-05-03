import pandas as pd
import numpy as np
import cv2


def clahe(img, clip=2, tile=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def eq_hist(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def rescale(img):
    return img * 1. / 255

