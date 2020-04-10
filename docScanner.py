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