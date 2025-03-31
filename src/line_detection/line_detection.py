import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from itertools import combinations

class LineDetector:
    def __init__(self, width=848, height=480, scale=1):
        self.width = width
        self.height = height
        self.scale = scale

    def filter_contours(self, img):
        dilfactor = 2
        dilationkernel = np.ones((dilfactor, dilfactor), np.uint8) 
        img_dil = cv2.dilate(img, dilationkernel, iterations=1)

        contours = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        for cnt in contours:
            x1, y1, w, h = cv2.boundingRect(cnt)
            w = max(w, int(20 * self.scale))
            h = max(h, int(20 * self.scale))
            rect = img_dil[y1:y1+h, x1:x1+w]
            
            th = int(self.scale * 50)
            if w < th and h < th:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
                
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        return img_masked

    def get_lines(self, img):
        sigma = 5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (sigma, sigma), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges
