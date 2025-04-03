import cv2
import numpy as np
from shapely.geometry import LineString
from itertools import combinations
from constants import width, scale
from image_worker import getColorMask, getRoiMask, filterWhite, filterContours

def getLines(img):
    """
    Detect lines in the given image using Canny edge detection and Hough transform.
    """
    sigma = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (sigma, sigma), 0)
    edges = cv2.Canny(blur, 50, 150)

    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(imghsv, (sigma, sigma), 0)
    maskColor = getColorMask(imghsv)
    blurMaskColor = getColorMask(blurhsv)

    maskRoi = getRoiMask(img)
    img_masked = cv2.bitwise_and(edges, maskRoi)
    img_masked = cv2.bitwise_and(img_masked, maskColor)

    img_masked = filterWhite(img_masked)
    img_masked = filterContours(img_masked)

    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8)
    img_masked = cv2.dilate(img_masked, dilationkernel, iterations=1)

    lines = cv2.HoughLinesP(
        img_masked, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 70,
        maxLineGap=int(scale * 10), minLineLength=int(scale * 20)
    )
    return lines


def newLines(lines):
    """
    Cluster and combine lines into new representative lines.
    """
    nlines = []
    if lines is not None:
        clusters = clusterLines(lines, int(scale * 10), 15)
        for cluster in clusters:
            newline = combineLines(cluster)
            nlines.append(newline)
        return nlines
    return 0


def splitLines(lines):
    """
    Split lines into left and right based on their slope.
    """
    llines = []
    rlines = []
    for line in lines:
        x1, y1, x2, y2 = line
        linepar = np.polyfit((x1, x2), (y1, y2), 1)  # slope and y-intercept
        angle = (180 / np.pi) * np.arctan(linepar[0])
        if angle > 5:
            rlines.append(line)
        if angle < -5:
            llines.append(line)
    return llines, rlines


def clusterLines(lines, th_dis, th_ang):
    """
    Cluster lines based on their proximity and angular similarity.
    """
    if lines is not None:
        cluster_total = 0
        cluster_id = np.zeros(len(lines), dtype=int)

        for i, j in combinations(enumerate(lines), 2):
            x1i, y1i, x2i, y2i = i[1][0]
            l1 = LineString([(x1i, y1i), (x2i, y2i)])
            x1j, y1j, x2j, y2j = j[1][0]
            l2 = LineString([(x1j, y1j), (x2j, y2j)])
            distance = l1.distance(l2)

            linepar1 = np.polyfit((x1i, x2i), (y1i, y2i), 1)
            angdeg1 = (180 / np.pi) * np.arctan(linepar1[0])
            linepar2 = np.polyfit((x1j, x2j), (y1j, y2j), 1)
            angdeg2 = (180 / np.pi) * np.arctan(linepar2[0])
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
    return 0


def combineLines(lines, hue=50):
    """
    Combine multiple lines into a single representative line.
    """
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
        angdeg = (180 / np.pi) * np.arctan(linepar[0])
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


def longestLine(lines):
    """
    Find the longest line from a list of lines.
    """
    longest = 0
    longestline = None
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((abs(x2 - x1))**2 + (abs(y2 - y1))**2)
        if length > longest:
            longest = length
            longestline = line
    return longestline


def findTarget(llines, rlines, horizonh, img, wl=1, wr=1, weight=1, bias=0, draw=1):
    """
    Find the target point for steering based on left and right lines.
    """
    drawimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if not llines and not rlines:
        target = False
    elif not rlines:
        lline = longestLine(llines)
        x1l, y1l, x2l, y2l = lline
        lineparL = np.polyfit((x1l, x2l), (y1l, y2l), 1)
        horizonxL = round((horizonh - lineparL[1]) / lineparL[0])
        if draw == 1:
            cv2.line(drawimg, (x1l, y1l), (x2l, y2l), (50, 200, 200), 3)
            cv2.circle(drawimg, (horizonxL, horizonh), 1, (50, 200, 200), 3)
        target = horizonxL
    elif not llines:
        rline = longestLine(rlines)
        x1r, y1r, x2r, y2r = rline
        lineparR = np.polyfit((x1r, x2r), (y1r, y2r), 1)
        horizonxR = round((horizonh - lineparR[1]) / lineparR[0])
        if draw == 1:
            cv2.line(drawimg, (x1r, y1r), (x2r, y2r), (100, 200, 200), 3)
            cv2.circle(drawimg, (horizonxR, horizonh), 1, (100, 200, 200), 3)
        target = horizonxR
    else:
        lline = longestLine(llines)
        rline = longestLine(rlines)
        x1r, y1r, x2r, y2r = rline
        x1l, y1l, x2l, y2l = lline
        lineparR = np.polyfit((x1r, x2r), (y1r, y2r), 1)
        horizonxR = round((horizonh - lineparR[1]) / lineparR[0])
        lineparL = np.polyfit((x1l, x2l), (y1l, y2l), 1)
        horizonxL = round((horizonh - lineparL[1]) / lineparL[0])
        heightL = lineparL[1]
        heightR = lineparR[0] * width + lineparR[1]
        target = ((horizonxL + horizonxR) / 2) + (heightL - heightR) * weight + bias
        if draw == 1:
            cv2.line(drawimg, (x1r, y1r), (x2r, y2r), (100, 200, 200), 3)
            cv2.line(drawimg, (x1l, y1l), (x2l, y2l), (50, 200, 200), 3)
            cv2.circle(drawimg, (int(target), horizonh), 1, (180, 200, 200), 3)
    return target