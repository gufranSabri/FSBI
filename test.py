import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage import filters

def get_dwt_rgb(img):
    b, g, r = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(img_gray, 'sym2', mode='reflect')
    cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, 'sym2', mode='reflect')
    cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, 'sym2', mode='reflect')

    # cA_r = cv2.resize(cA_r, (700,700), interpolation=cv2.INTER_LINEAR)
    # cA_g = cv2.resize(cA_g, (700,700), interpolation=cv2.INTER_LINEAR)
    # cA_b = cv2.resize(cA_b, (700,700), interpolation=cv2.INTER_LINEAR)

    # cA_r = (cA_r + r)/2
    # cA_g = (cA_g + g)/2
    # cA_b = (cA_b + b)/2

    # img_dwt = np.array([cA_b, cA_g, cA_r])
    # img_dwt = np.transpose(img_dwt, (1,2,0))

    # print(img_dwt)

    return cA_r / 290

image_path = "/Users/gufran/Developer/Projects/AI/DeepfakeDetection/testImg.png"
original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (700,700), interpolation=cv2.INTER_LINEAR)

res = get_dwt_rgb(original_image)
cv2.imshow("img", original_image)
cv2.imshow("img2", res)
cv2.waitKey(0)

