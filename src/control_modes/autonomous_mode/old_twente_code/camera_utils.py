import cv2
import os
from constants import height, width, FALLBACK_VIDEO_PATH
from line_detection import getLines, newLines, splitLines, longestLine
import numpy as np

def initialize_cameras():
    """
    Initialize the opencv camera capture devices. If no camera config is found or
    if cameras fail to open, fall back to using a sample mp4 video.
    """
    cameras = {}
    fallback_video_path = FALLBACK_VIDEO_PATH

    def load_fallback():
        print('[WARNING] No valid video configuration found. Falling back to MP4 video file.')
        if not os.path.exists(fallback_video_path):
            print(f"[ERROR] Fallback video {fallback_video_path} does not exist.")
            exit(1)
        capture = cv2.VideoCapture(fallback_video_path)
        if not capture.isOpened():
            print(f"[ERROR] Could not open fallback video {fallback_video_path}.")
            exit(1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return {"front": capture}

    # Add camera initialization logic here if needed
    return load_fallback()

def getHorizon(img):
    """
    Detect the horizon in the given image by analyzing lines.
    """
    lines = getLines(img)
    if lines is not None:
        lines = newLines(lines)
        llines, rlines = splitLines(lines)
        if not llines and not rlines:
            print("MISSING BOTH LINES")
        elif not rlines:
            print("MISSING RIGHT LINES")
        elif not llines:
            print("MISSING LEFT LINES")
        else:
            lline = longestLine(llines)
            rline = longestLine(rlines)
            x1r, y1r, x2r, y2r = rline
            x1l, y1l, x2l, y2l = lline
            lineparR = np.polyfit((x1r, x2r), (y1r, y2r), 1)
            lineparL = np.polyfit((x1l, x2l), (y1l, y2l), 1)
            x_h = (lineparR[1] - lineparL[1]) / (lineparL[0] - lineparR[0])
            y_h = x_h * lineparL[0] + lineparL[1]
            return round(x_h), round(y_h)
    print("HORIZON NOT FOUND DUE TO NO LINES DETECTED")
    return 0