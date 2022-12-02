import os
import cv2
import sys
import numpy as np

sys.path.append("bin/Release")
os.environ['path'] += os.path.abspath("bin/Release")
import Pyimage

# libimage.dll 用的是 CHW 格式, opencv 用的是 HWC 格式
def convert_to_opencv_format(image):
    h, w, c = image.shape
    cvImage = np.zeros(shape=(h,w,c))
    image_data = image.flatten()

    for k in range(c):
        for j in range(h):
            for i in range(w):
                index = i + w*j + w*h*k
                cvImage[j,i,k] = image_data[index]

    cvImage = cvImage.astype(np.float32)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_RGB2BGR)
    return cvImage

image_path = "data/Rainier3.png"
image = Pyimage.load_image(image_path, 3)
h, w, c = image.shape
print(image)

Pyimage.save_image("data/Rainier3.save.py", image, 0)

resize_image = Pyimage.resize(image, w*2, h*2)
Pyimage.save_image("data/Rainier3.resize.py", resize_image, 1)

cvImage = convert_to_opencv_format(image)
cv2.imwrite("data/Rainier3.save.opencv.png", cvImage)

