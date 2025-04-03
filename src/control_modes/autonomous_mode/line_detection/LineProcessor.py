# line_processing.py
import numpy as np
from shapely.geometry import LineString
from itertools import combinations

from src.control_modes.autonomous_mode.line_detection.utils import getColorMask, getRoiMask, filterContours, filterWhite
import cv2

def clusterLines(lines, th_dis, th_ang):
    if lines is None:
        return 0
        
    cluster_total = 0
    cluster_id = np.zeros(len(lines), dtype=int)

    for i, j in combinations(enumerate(lines), 2):
        x1i, y1i, x2i, y2i = i[1][0]
        l1 = LineString([(x1i, y1i), (x2i, y2i)])
        x1j, y1j, x2j, y2j = j[1][0]
        l2 = LineString([(x1j, y1j), (x2j, y2j)])
        
        distance = l1.distance(l2)
        linepar1 = np.polyfit((x1i, x2i), (y1i, y2i), 1)
        angdeg1 = (180/np.pi) * np.arctan(linepar1[0])
        linepar2 = np.polyfit((x1j, x2j), (y1j, y2j), 1)
        angdeg2 = (180/np.pi) * np.arctan(linepar2[0])
        angdif = abs(angdeg1 - angdeg2)
        
        if distance < th_dis and angdif < th_ang:
            if cluster_id[i[0]] == 0 and cluster_id[j[0]] == 0:
                cluster_total += 1
                cluster_id[i[0]] = cluster_total
                cluster_id[j[0]] = cluster_total
            elif cluster_id[j[0]] == 0:
                cluster_id[j[0]] = cluster_id[i[0]]
    
    for count, id in enumerate(cluster_id):
        if id == 0:
            cluster_total += 1
            cluster_id[count] = cluster_total
    
    cluster_id = cluster_id - 1
    clusters = [[] for _ in range(cluster_total)]
    for i, line in enumerate(lines):
        clusters[cluster_id[i]].append(line)

    return clusters

def combineLines(lines):
    x1a = []
    x2a = []
    y1a = []
    y2a = []
    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1a.append(x1)
        x2a.append(x2)
        y1a.append(y1)
        y2a.append(y2)
        linepar = np.polyfit((x1, x2), (y1, y2), 1)
        angdeg = (180/np.pi) * np.arctan(linepar[0])
        angles.append(angdeg)
        
    ang = np.mean(angles)
    x1 = min(x1a)
    x2 = max(x2a)
    
    if ang > 0:
        y1 = min(y1a)
        y2 = max(y2a)
    else:
        y1 = max(y1a)
        y2 = min(y2a)
    
    line_new = np.array([x1, y1, x2, y2])
    return line_new

def splitLines(lines):
    llines = []
    rlines = []
    for line in lines:
        x1, y1, x2, y2 = line
        linepar = np.polyfit((x1, x2), (y1, y2), 1)
        angle = (180/np.pi) * np.arctan(linepar[0])
        if angle > 5:
            rlines.append(line)
        if angle < -5:
            llines.append(line)
    return llines, rlines


def getLines(img, scale, height, width):
    """Detect lines in the image."""
    sigma = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (sigma, sigma), 0)
    edges = cv2.Canny(blur, 50, 150)

    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    maskColor = getColorMask(imghsv)
    blurMaskColor = getColorMask(blurhsv)

    maskRoi = getRoiMask(img, scale)
    img_masked = cv2.bitwise_and(edges, maskRoi)
    img_masked = cv2.bitwise_and(img_masked, maskColor)

    img_masked = filterWhite(img_masked, height, width)
    img_masked = filterContours(img_masked, scale)

    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    img_masked = cv2.dilate(img_masked, dilationkernel, iterations=1)

    lines = cv2.HoughLinesP(img_masked, cv2.HOUGH_PROBABILISTIC, np.pi/180, 70, maxLineGap=int(scale * 10), minLineLength=int(scale * 20))
    return lines