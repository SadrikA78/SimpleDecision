from PIL import Image
import pytesseract
import cv2
import os
import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
import argparse
import imutils
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

image = 'img2.png'

# preprocess = "thresh"

# # загрузить образ и преобразовать его в оттенки серого
# image = cv2.imread(image)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # проверьте, следует ли применять пороговое значение для предварительной обработки изображения

# if preprocess == "thresh":
    # gray = cv2.threshold(gray, 0, 255,
        # cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # если нужно медианное размытие, чтобы удалить шум
# elif preprocess == "blur":
    # gray = cv2.medianBlur(gray, 3)

# # сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR

# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)
# text = pytesseract.image_to_string(Image.open(filename))
# os.remove(filename)
#print(pytesseract.image_to_string(Image.open(image), lang="eng"))




# image = cv2.imread(image)
# h, w, c = image.shape
# boxes = pytesseract.image_to_boxes(image) 
# for b in boxes.splitlines():
    # b = b.split(' ')
    # image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# b,g,r = cv2.split(image)
# rgb_img = cv2.merge([r,g,b])

# plt.figure(figsize=(16,12))
# plt.imshow(rgb_img)
# plt.title('SAMPLE INVOICE WITH CHARACTER LEVEL BOXES')
# plt.show()
# custom_config = r'--oem 3 --psm 6'
# print(pytesseract.image_to_string(image, config=custom_config))


img = cv2.imread('img1.png')
# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# keys = list(d.keys())

# date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'

# n_boxes = len(d['text'])
# for i in range(n_boxes):
    # if int(d['conf'][i]) > 60:
    	# if re.match(date_pattern, d['text'][i]):
	        # (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)

img = cv2.imread('img2.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make a copy of the original image
cimg = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
img = cv2.medianBlur(img, 5)
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9, 
                            minDist=80, param1=110, param2=39, maxRadius=70)
try:
    for co, i in enumerate(circles[0, :], start=1):
        # draw the outer circle in green
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle in red
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
except TypeError:
    print ('-')
# for co, i in enumerate(circles[0, :], start=1):
    # cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
print("Найдено печатей:", co)