import numpy as np
import cv2
from skimage.filters import threshold_local
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog

def order_points(pts):
	rectangle = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rectangle[0] = pts[np.argmin(s)]
	rectangle[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rectangle[1] = pts[np.argmin(diff)]
	rectangle[3] = pts[np.argmax(diff)]
	return rectangle

def fp_transform(img, pts):
	rectangle = order_points(pts)
	(tl, tr, br, bl) = rectangle
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))