import threading
import os
import cv2
from queue import Queue
import numpy as np
from PIL import Image, ImageDraw

class ImageWorker:
    """
    A worker that writes images to disk.
    """

    def __init__(self, image_queue: Queue, folder: str):
        self.queue = image_queue
        self.thread = threading.Thread(target=self._process, args=(), daemon=True)
        self.folder: str = folder

    def start(self):
        self.thread.start()

    def stop(self):
        self.queue.join()

    def put(self, data):
        self.queue.put(data)

    def _process(self):
        while True:
            filename, image_type, image = self.queue.get()
            cv2.imwrite(os.path.join(self.folder, image_type, f'{filename}.png'), image)
            self.queue.task_done()

def getColorMask(imghsv):
    """
    Generate a color mask for the given HSV image.
    """
    sigma = 3  # Blurring constant
    lower_range = (0, 0, 160)  # Lower range of red color in HSV
    upper_range = (255, 255, 255)  # Upper range of red color in HSV
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    maskhsv = cv2.inRange(blurhsv, lower_range, upper_range)

    # Dilation
    dilfactor = 4
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    maskhsvdil = cv2.dilate(maskhsv, dilationkernel, iterations=2)
    return maskhsvdil


def getRoiMask(img):
    """
    Generate a region of interest (ROI) mask for the given image.
    """
    width = img.shape[1]
    height = img.shape[0]
    mid = width / 2
    maskwd = 0  # Left and right margin
    maskwu = mid  # Width of the upper part of the mask
    maskh = 220  # Height of the mask
    polygon = [
        (maskwd, height),
        (mid - maskwu, maskh),
        (mid + maskwu, maskh),
        (width - maskwd, height),
    ]
    imgmask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=255)
    mask = np.array(imgmask)
    return mask


def filterWhite(img_masked):
    """
    Filter out white regions in the given masked image.
    """
    window_size = 24
    hstart = 240
    sidemargin = 4
    mask = np.ones_like(img_masked) * 255

    for row in range(round((img_masked.shape[0] - hstart) / window_size)):
        for col in range(round((img_masked.shape[1] - 2 * sidemargin) / window_size)):
            window = img_masked[
                     row * window_size + hstart : row * window_size + window_size + hstart,
                     col * window_size + sidemargin : col * window_size + window_size + sidemargin,
                     ]
            density = np.mean(window)
            if density > 45:
                mask[
                row * window_size + hstart : row * window_size + window_size + hstart,
                col * window_size + sidemargin : col * window_size + window_size + sidemargin,
                ] = np.zeros([window_size, window_size])

    img_masked = cv2.bitwise_and(img_masked, mask)
    return img_masked


def filterContours(img):
    """
    Filter out small contours in the given image.
    """
    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    img_dil = cv2.dilate(img, dilationkernel, iterations=1)

    contours = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        x1, y1, w, h = cv2.boundingRect(cnt)
        w = max(w, 20)
        h = max(h, 20)
        rect = img_dil[y1 : y1 + h, x1 : x1 + w]
        density = np.mean(rect)
        th = 50
        if w < th and h < th:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    img_masked = cv2.bitwise_and(img, img, mask=mask)
    return img_masked