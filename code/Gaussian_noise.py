import numpy as np
import cv2
import matplotlib.pyplot as plt

def noiseElimination(img_gray):
    h, templateWindowSize, searchWindowSize = 15.0, 7, 21
    dst = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    return dst

'img = cv2.imread("exp1.jpg")

'''b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])


h, hColor, templateWindowSize, searchWindowSize = 3, 10, 7, 21
dst = cv2.fastNlMeansDenoisingColored(rgb_img, None, h, hColor, templateWindowSize, searchWindowSize)

b,g,r = cv2.split(dst)
rgb_dst = cv2.merge([r,g,b])

plt.subplot(211),plt.imshow(rgb_img)
plt.subplot(212),plt.imshow(rgb_dst)
plt.show()'''

'''plt.subplot(211),plt.imshow(img)
plt.subplot(212),plt.imshow(noiseElimination(img))
plt.show()'''

'print(noiseElimination(img).shape)