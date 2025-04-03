# image_utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw

def getColorMask(imghsv):
    sigma = 3
    lower_range = (0, 0, 160)
    upper_range = (255, 255, 255)
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    maskhsv = cv2.inRange(blurhsv, lower_range, upper_range)
    
    dilfactor = 4
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8) 
    maskhsvdil = cv2.dilate(maskhsv, dilationkernel, iterations=2)
    return maskhsvdil

def getRoiMask(img, scale):
    width = img.shape[1]
    height = img.shape[0]
    mid = width / 2
    maskwd = 0
    maskwu = mid
    maskh = 220 * scale
    
    polygon = [(maskwd, height), (mid - maskwu, maskh), (mid + maskwu, maskh), (width - maskwd, height)]
    imgmask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=255)
    mask = np.array(imgmask)
    return mask

def filterContours(img, scale):
    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8) 
    img_dil = cv2.dilate(img, dilationkernel, iterations=1)

    contours = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        x1, y1, w, h = cv2.boundingRect(cnt)
        w = max(w, int(20 * scale))
        h = max(h, int(20 * scale))
        rect = img_dil[y1:y1+h, x1:x1+w]
        
        th = int(scale * 50)
        if w < th and h < th:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
            
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    return img_masked

def filterWhite(img_masked, height, width):
    window_size = 24
    hstart = 240
    sidemargin = 4
    mask = np.ones_like(img_masked)
    mask = mask * 255
    
    for row in range(round((height - hstart) / window_size)):
        for col in range(round((width - 2 * sidemargin) / window_size)):
            window = img_masked[row * window_size + hstart:row * window_size + window_size + hstart,
                                 col * window_size + sidemargin:col * window_size + window_size + sidemargin]
            density = np.mean(window)
            if density > 45:
                mask[row * window_size + hstart:row * window_size + window_size + hstart,
                     col * window_size + sidemargin:col * window_size + window_size + sidemargin] = np.zeros([window_size, window_size])
    
    img_masked = cv2.bitwise_and(img_masked, mask)
    return img_masked

