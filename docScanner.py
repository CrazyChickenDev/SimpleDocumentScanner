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
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rectangle, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped


root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()
image = cv2.imread(image_path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
	


try:
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	warped = fp_transform(orig, screenCnt.reshape(4, 2) * ratio)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255
	cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	cv2.imwrite('Scanned.jpg',warped)

	cv2.waitKey(0)
except NameError:
	print('Image not Clear')

 	
