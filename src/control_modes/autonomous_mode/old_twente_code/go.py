import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from itertools import combinations
from time import sleep

#import onnxruntime as rt
import time
import can
import struct

from typing import Optional, Dict, List

import sys
import math

#for recording:
from datetime import datetime
from queue import Queue

#object detection
import shutil
import json
import torch

import platform
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
import video as video

import threading
from collections import deque
import collections
import csv
import psutil

CAN_MSG_SENDING_SPEED = .040
height = 480
width = 848
scale = 1
OBJECT_FPS_CAP = 20
OBJECT_DOWNSAMPLE_FACTOR =1
 
class CanListener:
    """
    A can listener that listens for specific messages and stores their latest values.
    """

    _id_conversion = {
        0x110: 'brake',
        0x220: 'steering',
        0x330: 'throttle',
        0x440: 'speed_sensor',
        0x1e5: 'steering_sensor'
    }

    def __init__(self, bus: can.Bus):
        self.bus = bus
        self.thread = threading.Thread(target = self._listen, args = (), daemon = True)
        self.running = False
        self.data : Dict[str, List[int]] = {name: None for name in self._id_conversion.values()}
    
    def start_listening(self):
        self.running = True
        self.thread.start()
    
    def stop_listening(self):
        self.running = False
    
    def get_new_values(self):
        values = self.data
        return values

    def _listen(self):
        while self.running:
            message: Optional[can.Message] = self.bus.recv(.5)
            if message and message.arbitration_id in self._id_conversion:
                self.data[self._id_conversion[message.arbitration_id]] = message.data

class ImageWorker:
    """
    A worker that writes images to disk.
    """

    def __init__(self, image_queue: Queue, folder: str):
        self.queue = image_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
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

class CanWorker:
    """
    A worker that writes can-message values to disk.
    """

    def __init__(self, can_queue: Queue, folder: str):
        self.queue = can_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
        self.folder_name = folder
        self.file_pointer = open(os.path.join(self.folder_name, f'recording.csv'), 'w')
        print('Timestamp|Steering|SteeringSpeed|Throttle|Brake|SpeedSensor|SteeringSensor', file = self.file_pointer)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.queue.join()
        self.file_pointer.close()
    
    def put(self, data):
        self.queue.put(data)
    
    def _process(self):
        while True:
            timestamp, values = self.queue.get()
            steering = str(struct.unpack("f", bytearray(values["steering"][:4]))[0]) if values["steering"] else ""
            steering_speed = str(struct.unpack(">I", bytearray(values["steering"][4:]))[0]) if values["steering"] else ""
            throttle = str(values["throttle"][0]/100) if values["throttle"] else ""
            brake = str(values["brake"][0]/100) if values["brake"] else ""
            #speed_sensor = ""
            #speed_sensor = str(values["speed_sensor"][0]) if values["speed_sensor"] else ""
#            speed_sensor = str(struct.unpack(">H", bytearray(values["speed_sensor"][:2]))[0]) if values["speed_sensor"] else ""
            speed_sensor = str(struct.unpack(">H", bytearray(values["speed_sensor"][:2]))[0]) if values["speed_sensor"] else ""
            if values["steering_sensor"]:
                steering_sensor = (values["steering_sensor"][1] << 8 | values["steering_sensor"][2])
                steering_sensor -= 65536 if steering_sensor > 32767 else 0
            else:
                steering_sensor = ""
            print(f'{timestamp}|{steering}|{steering_speed}|{throttle}|{brake}|{speed_sensor}|{steering_sensor}', file=self.file_pointer)
            self.queue.task_done()



def setExposure(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    count = 0
    max_count = 10
    while(True):
        ret, frame0 = cap.read()
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        #print("count:",count)
        #print("Exposure set to:", exposure)
        count = count + 1
        if count >= max_count:
            break
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    print("Exposure set to:", exposure)
    return exposure

def filterContours(img):
    dilfactor = 2
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8) 
    img_dil = cv2.dilate(img, dilationkernel, iterations=1)

    #cv2.imshow("test", img_dil)
    contours = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse= True)
    #print("DIL", img_dil.shape)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        
        #area= cv2.contourArea(cnt)
        x1,y1,w,h = cv2.boundingRect(cnt)
        #rect_area = w*h
        w = max(w,int(20*scale))
        h = max(h,int(20*scale))
        #rect = img_dil[x1:x1+w,y1:y1+h]
        rect = img_dil[y1:y1+h,x1:x1+w]
        #print("RECT", rect.shape, "\twidth,height:", w, h, "\t mean:", np.mean(rect))
        density = np.mean(rect)
        th = int(scale*50)
        if w<th and h<th:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
        #if density > 60:
        #    cv2.drawContours(mask, [cnt], -1, 0, -1)
    #if h>0 and w>0:
        #cv2.imshow("RECT", rect)
        #q = 0
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    
    #cv2.imshow("masked_function", img_masked)
    return img_masked

def clusterLines(lines, th_dis, th_ang):
    #print('\nSTART CLUSTER\n')
    if lines is not None:
        #lines = sorted(lines)
        cluster_total = 0
        cluster_id = np.zeros(len(lines), dtype=int)

        #print(len(lines))
        for i,j in combinations(enumerate(lines),2):
            
            #print("i,j:",i[1],j)
            
            x1i, y1i, x2i, y2i = i[1][0]
            l1 = LineString([(x1i, y1i), (x2i, y2i)])
            x1j, y1j, x2j, y2j = j[1][0]
            l2 = LineString([(x1j, y1j), (x2j, y2j)])
            distance = l1.distance(l2)
            linepar1= np.polyfit((x1i,x2i),(y1i,y2i),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
            angdeg1 = (180/np.pi)*np.arctan(linepar1[0])
            linepar2= np.polyfit((x1j,x2j),(y1j,y2j),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
            angdeg2 = (180/np.pi)*np.arctan(linepar2[0])
            angdif = abs(angdeg1 - angdeg2)
            
            
            if distance < th_dis and angdif < th_ang:
                #print("i,j:",i,j)
                #print("distance:", distance, "angdif:",angdif)
                #print("cluster_ids:", cluster_id[i[0]], cluster_id[j[0]])
                if cluster_id[i[0]] == 0 and cluster_id[j[0]] == 0:
                    cluster_total += 1
                    cluster_id[i[0]] = cluster_total
                    cluster_id[j[0]] = cluster_total
                elif cluster_id[j[0]] == 0:
                    cluster_id[j[0]] = cluster_id[i[0]]
        
        #Give ids to lines that were not in pairs:
        #print("cluster_id", cluster_id)       
        for count, id in enumerate(cluster_id):
            if id == 0:
                cluster_total += 1
                cluster_id[count] = cluster_total
        cluster_id = cluster_id - 1
        #print("cluster_id", cluster_id)
        #print("cluster_total:",cluster_total)
        #clusters = [None] * cluster_total
        clusters = [[] for _ in range(cluster_total)]
        for i, line in enumerate(lines):
            clusters[cluster_id[i]].append(line)
            #print("clusterslen:", len(clusters))
        #print("FINAL clusters:", clusters)


        return clusters
    return 0

def combineLines(lines, hue = 50):
        
        x1a = []
        x2a = []
        y1a = []
        y2a = []
        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1a.append(x1); x2a.append(x2); y1a.append(y1); y2a.append(y2)
            linepar= np.polyfit((x1,x2),(y1,y2),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
            angdeg = (180/np.pi)*np.arctan(linepar[0])
            angles.append(angdeg)
        ang = np.mean(angles)
        #print("ang:",ang)
        x1 = min(x1a)
        x2 = max(x2a)
        if ang > 0:
            y1 = min(y1a)
            y2 = max(y2a)
        else:
            y1 = max(y1a)
            y2 = min(y2a)
        
        """
        linepar= np.polyfit((x1,x2),(y1,y2),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        angle = (180/np.pi)*np.arctan(linepar[0])
        if abs(angle) <5:
            hue = 180
            cv2.line(img, (x1,y1),(x2,y2), (hue,200,200), 3)
        elif angle > 0:
            hue = 100
            cv2.line(img, (x1,y1),(x2,y2), (hue,200,200), 3)
        else:
            hue = 50
            cv2.line(img, (x1,y1),(x2,y2), (hue,200,200), 3)
        #cv2.line(img, (x1,y1),(x2,y2), (hue,200,200), 3)
        """
        line_new = np.array([x1,y1,x2,y2])
        #print("newline", line_new)
        return line_new

def newLines(lines):
    nlines = []
   
    if lines is not None:
        clusters = clusterLines(lines, int(scale*10), 15)
        for cluster in clusters:
            newline = combineLines(cluster)
            nlines.append(newline)
        return nlines
    return 0 

def splitLines(lines):
    llines = []
    rlines = []
    for line in lines:
        x1, y1, x2, y2 = line
        linepar= np.polyfit((x1,x2),(y1,y2),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        angle = (180/np.pi)*np.arctan(linepar[0])
        if angle > 5:
            rlines.append(line)
        if angle < -5:
            llines.append(line)
    return llines, rlines

def getRoiMask(img):
    width = img.shape[1]
    height = img.shape[0]
    mid = width/2
    maskwd = 0#-1000
    maskwu = mid#65
    maskh = 220 * scale#180 #180 og value
    polygon = [(maskwd,height),(mid - maskwu, maskh),(mid + maskwu,maskh),(width-maskwd,height)]#,(width-100,height),(width - 200, height-100),(200,height-100),(100,height)]
    imgmask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=255)
    mask = np.array(imgmask)
    return mask

def getColorMask(imghsv):
    sigma = 3 #blurring constant
    lower_range = (0, 0, 160) # lower range of red color in HSV
    upper_range = (255, 255, 255) # upper range of red color in HSV
    blurhsv = cv2.GaussianBlur(imghsv,(sigma,sigma),0)
    maskhsv = cv2.inRange(blurhsv, lower_range, upper_range)
    
    #dilation
    dilfactor = 4
    dilationkernel = np.ones((dilfactor, dilfactor), np.uint8) 
    maskhsvdil = cv2.dilate(maskhsv, dilationkernel, iterations=2)
    return maskhsvdil

def getMatrix(path):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    square_size = 25 #size of a single chessboard square in mm
    objp = objp * square_size
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    allCorners = [] # 2d points in image plane.
    #print("objp:", objp)
    filenames = next(os.walk(path), (None, None, []))[2]  # [] if no file
    for file in filenames:    
        img_name = path+file
        #print(img_name)
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,6))#, None)
        
        if ret == True:
            objpoints.append(objp)  
            subPixCorners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            allCorners.append(subPixCorners)
            #cv2.drawChessboardCorners(img, (8,6), subPixCorners, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

        else:
            print("image", img_name, "was returned with ret="+ ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, allCorners, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

def longestLine(lines):
    longest = 0
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((abs(x2-x1))**2+(abs(y2-y1))**2)
        if length > longest:
            longest = length
            longestline = line
    return longestline

def filterWhite(img_masked):
    #white squares
    window_size = 24
    hstart = 240
    sidemargin = 4
    mask = np.ones_like(img_masked)
    mask = mask * 255
  
    for row in range(round((height-hstart)/window_size)):
       for col in range(round((width-2*sidemargin)/window_size)):
            window = img_masked[row*window_size+hstart:row*window_size+window_size+hstart,col*window_size+sidemargin:col*window_size+window_size+sidemargin]
            density = np.mean(window)
            if density > 45:
                mask[row*window_size+hstart:row*window_size+window_size+hstart,col*window_size+sidemargin:col*window_size+window_size+sidemargin]=np.zeros([window_size,window_size]) 
            #print(density)
            #print(row*window_size+hstart, col*window_size, img_masked.shape, window.shape)
            #print(row+hstart, row+hstart+window_size)
            #sample = img_masked[row+hstart]
    #cv2.imshow("mask", mask)
    img_masked = cv2.bitwise_and(img_masked, mask)
    #cv2.imshow("fil", img_masked)
    return img_masked

def getLines(img):
    sigma = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(sigma,sigma),0)
    edges = cv2.Canny(blur,50,150)


    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(imghsv,(sigma,sigma),0)
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
    
    lines = cv2.HoughLinesP(img_masked, cv2.HOUGH_PROBABILISTIC, np.pi/180, 70,maxLineGap = int(scale*10), minLineLength = int(scale*20))
    return lines

def findTarget(llines, rlines, horizonh, img, wl = 1, wr = 1, weight = 1, bias = 0, draw = 1): #returns False if non found, otherwise returns target x
    drawimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if not llines and not rlines:#rlines is not None and llines is not None:
        target = False
    elif not rlines:
        #print("LEFT", wl, wr)
        lline = longestLine(llines)
        x1l, y1l, x2l, y2l = lline
         
        lineparL= np.polyfit((x1l,x2l),(y1l,y2l),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        horizonxL = round((horizonh-lineparL[1])/lineparL[0])
        
        if draw == 1:
            cv2.line(drawimg, (x1l,y1l),(x2l,y2l), (50,200,200), 3) 
            cv2.circle(drawimg,(horizonxL,horizonh), 1, (50,200,200), 3)
        
        target = horizonxL
        #Error = horizonxL - width/2
        #print("error:", Error)


    elif not llines:
        #print("RIGHT", wl, wr)
        rline = longestLine(rlines)

        x1r, y1r, x2r, y2r = rline
        lineparR= np.polyfit((x1r,x2r),(y1r,y2r),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        horizonxR = round((horizonh-lineparR[1])/lineparR[0])
        
        if draw == 1:
            cv2.line(drawimg, (x1r,y1r),(x2r,y2r), (100,200,200), 3)
            cv2.circle(drawimg,(horizonxR,horizonh), 1, (100,200,200), 3)
        target = horizonxR
        #Error = horizonxR - width/2
        #print("error:", Error)

    else:
        #print("BOTH", wl ,wr)
        lline = longestLine(llines)
        rline = longestLine(rlines)

        x1r, y1r, x2r, y2r = rline
        x1l, y1l, x2l, y2l = lline
    
        lineparR= np.polyfit((x1r,x2r),(y1r,y2r),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        horizonxR = round((horizonh-lineparR[1])/lineparR[0])
        
        lineparL= np.polyfit((x1l,x2l),(y1l,y2l),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
        horizonxL = round((horizonh-lineparL[1])/lineparL[0])

        #calculate intersections with borders
        heightL = lineparL[1]
        heightR = lineparR[0]*width + lineparR[1]

        
        
        x_h = (lineparR[1]-lineparL[1])/(lineparL[0]-lineparR[0])
        y_h = x_h * lineparL[0] + lineparL[1]


        wl = max(wl,0.01)
        wr = max(wr,0.01)
        #target = ((horizonxL*wl+horizonxR*wr)/(wl+wr))+(heightL-heightR)*weight + bias
        target = ((horizonxL+horizonxR)/(2))+(heightL-heightR)*weight + bias

        #print("target\t", round(target-424), "\twl\t", wl, "\twr\t", wr)
        #target = ((horizonxL*wl+horizonxR*wr)/(2))+(heightL-heightR)*weight + bias
        #print("heights:\t\t\t", round(heightL), "\t", round(heightR), "\tdifference:", round(heightL-heightR), "\tTarget:",target, "\twl,wr:", wl,'\t', wr)
        if draw == 1:
            cv2.line(drawimg, (x1r,y1r),(x2r,y2r), (100,200,200), 3)
            cv2.line(drawimg, (x1l,y1l),(x2l,y2l), (50,200,200), 3)  
            cv2.circle(drawimg,(round(x_h),round(y_h)), 1, (210,200,200), 3) 
            cv2.circle(drawimg,(horizonxR,horizonh), 1, (100,200,200), 3)  
            cv2.circle(drawimg,(horizonxL,horizonh), 1, (50,200,200), 3)
            cv2.circle(drawimg,(int(target),horizonh), 1, (180,200,200), 3)
        
        
 

    if draw == 1:
        cv2.circle(drawimg,(int(width/2),horizonh), 1, (0,0,255), 3)
        drawimg = cv2.cvtColor(drawimg, cv2.COLOR_HSV2BGR)
        cv2.cvtColor(drawimg, cv2.COLOR_HSV2BGR)

        # cv2.imshow("SHOW", drawimg)
    return target


def getHorizon(img):
    lines = getLines(img) 
    if lines is not None:
        lines = newLines(lines)
        llines, rlines = splitLines(lines)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        print("dimensions",img.shape)
        if not llines and not rlines:#rlines is not None and llines is not None:
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

            lineparR= np.polyfit((x1r,x2r),(y1r,y2r),1) #returns slope and y intercept(y coordinaat snijpunt y-as)
            lineparL= np.polyfit((x1l,x2l),(y1l,y2l),1) #returns slope and y intercept(y coordinaat snijpunt y-as)

            x_h = (lineparR[1]-lineparL[1])/(lineparL[0]-lineparR[0])
            y_h = x_h * lineparL[0] + lineparL[1]
        #print("SUCCES")
        
        return round(x_h), round(y_h)
    else:
        print("HORIZON NOT FOUND DUE TO NO LINES DETECTED")
    return 0

def initialize_cameras() -> Dict[str, cv2.VideoCapture]:
    """
    Initialize the opencv camera capture devices. If no camera config is found or
    if cameras fail to open, fall back to using a sample mp4 video.
    """
    config: video.CamConfig = video.get_camera_config()
    cameras: Dict[str, cv2.VideoCapture] = dict()
    fallback_video_path = "src/control_modes/autonomous_mode/old_twente_code/recording 13-06-2024 11-54-31.mp4"

    def load_fallback():
        print('[WARNING] No valid video configuration found. Falling back to MP4 video file.', file=sys.stderr)
        if not os.path.exists(fallback_video_path):
            print(f"[ERROR] Fallback video {fallback_video_path} does not exist.", file=sys.stderr)
            exit(1)
        capture = cv2.VideoCapture(fallback_video_path)
        if not capture.isOpened():
            print(f"[ERROR] Could not open fallback video {fallback_video_path}.", file=sys.stderr)
            exit(1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return {"front": capture}

    if not config:
        return load_fallback()

    for camera_type, path in config.items():
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            print(f"[WARNING] Camera {camera_type} at {path} could not be opened.", file=sys.stderr)
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FPS, 30)
        exposure = setExposure(capture)
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cameras[camera_type] = capture

    if not cameras:
        return load_fallback()

    return cameras


def initialize_can() -> can.Bus:
    """
    Set up the can bus interface and apply filters for the messages we're interested in.
    """
    bus = can.Bus(interface='virtual', channel='vcan0', bitrate=500000)
    bus.set_filters([
        {'can_id': 0x110, 'can_mask': 0xfff, 'extended': False}, # Brake
        {'can_id': 0x220, 'can_mask': 0xfff, 'extended': False}, # Steering
        {'can_id': 0x330, 'can_mask': 0xfff, 'extended': False}, # Throttle
        {'can_id': 0x440, 'can_mask': 0xfff, 'extended': False}, # Speed sensor
        {'can_id': 0x1e5, 'can_mask': 0xfff, 'extended': False}, # Steering sensor
    ])
    return bus

"""
All functions below are for the object detection part of the code
"""
def estimate_distance(x1, y1, x2, y2, real_width, real_height, x_offset=424, y_offset=240, image_width=848, image_height=480,fov_based = False):
    """
    estimate_distance:
    takes the box corner positions, the objects real width and height and other image variables to calculate the distance from the center of the 
    car to the object
    """
    # Constants for Logitech StreamCam
    camera_fov_h = 67.5  # Horizontal field of view in degrees
    camera_fov_v = 41.2  # Vertical field of view in degrees
    focal_length = 540   # focal length of the camera
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    
    #Standard distance offset (distance from front of car to camera)
    camera_offset  = 0.3
    #Assure there is no division by 0
    if box_height <= 0:
        box_height = 1 
    if box_width <=0:
        box_width = 1
    #FOV based calculations
    if fov_based:
        # Translate bounding box coordinates to full image coordinates
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
    
        # Calculate the center and dimensions of the bounding box
        box_x_center = (x1 + x2) / 2
        box_y_center = (y1 + y2) / 2
        
    
        # Calculate the horizontal and vertical angles relative to the center of the image
        angle_h = (box_x_center - image_width / 2) * (camera_fov_h / image_width)
        angle_v = (box_y_center - image_height / 2) * (camera_fov_v / image_height)
    
        # Calculate the estimated distances using width and height
        est_w_d = (real_width * focal_length) / box_width
        est_h_d = (real_height * focal_length) / box_height
    
        # Adjust the estimated distances based on the angles
        adjusted_est_w_d = est_w_d / math.cos(math.radians(angle_h))
        adjusted_est_h_d = est_h_d / math.cos(math.radians(angle_v))
    
        # Average the distances if the entire object is within the FOV
        if box_width < box_height:
            distance = (adjusted_est_w_d + adjusted_est_h_d) / 2
        else:
            distance = adjusted_est_w_d
    
        return max(distance -camera_offset,0)
    #focal based calculations
    else:
        
        est_w_d = (focal_length * real_width)/box_width
        est_h_d = (focal_length * real_height)/box_height
        
        return max((est_w_d+ est_h_d)/2 - camera_offset,0)



def load_model(weights, device):
    """
    This function loads up a pretrained model
    """
    model = DetectMultiBackend(weights, device=device, fp16=False) #False for CPU
    model.warmup()
    return model

def quantize_model(model):
    """
    This model turns the model parameters from 32 bits down to 8 bits to reduce the computation load
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def run_inference(model, frame, device, stride=32):
    """
    This function takes as input a frame and the model and gives back the predictions in yolo format 
    """
    # Resize the image to ensure its dimensions are multiples of the model's stride
    img_size = check_img_size(frame.shape[:2], s=stride)  # Ensure multiple of stride
    img = cv2.resize(frame, (img_size[1], img_size[0]))

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # Convert to [3, height, width]
    img = np.expand_dims(img, axis=0)  # Add batch dimension [1, 3, height, width]
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to 0.0 - 1.0

    with torch.no_grad():
        pred = model(img)
    return pred, img.shape

def process_detections(pred, frame, img_shape, conf_thres=0.70, iou_thres=0.45, max_det=1000):
    """
    This functions takes the model predictions, the image and its filter parameters and gives back the detections
    classes 
    """
    det = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]
    detections = []

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = conf.item()
            class_id = int(cls.item())
            if class_id == 0:
                obj_class = 'Car'
            elif class_id == 1:
                obj_class = 'Person'
            elif class_id == 2:
                obj_class = 'Speed-limit-10km-h'
            elif class_id == 3:
                obj_class = 'Speed-limit-15km-h'
            elif class_id == 4:
                obj_class = 'Speed-limit-20km-h'
            elif class_id == 5:
                obj_class = 'Traffic Light Green'
            elif class_id == 6:
                obj_class = 'Traffic Light Red'
            else:
                obj_class = 'Unknown'

            detections.append({
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

    return detections

def find_next_available_dir(base_path, base_name):
    """
    Finds the next available directory to have as name
    """
    counter = 0
    while True:
        new_path = os.path.join(base_path, f"{base_name}_{counter}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def clear_directory(directory):
    """
    clears a directory of images
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def initialize(weights_path, output_dir_base):
    """
    This function initialzes all the parts needed for object detection like the model and device (CPU)
    """
    # Base directory for saving outputs
    os.makedirs(output_dir_base, exist_ok=True)
    print("Output Directory Created")

    # Determine the next available directory for saving detections
    detections_dir = find_next_available_dir(output_dir_base, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    print(f"Detections Directory Created: {detections_dir}")

   # Load model once
    device = select_device('CPU')
    model = load_model(weights_path, device)
    
   
    quantized_model = quantize_model(model)
    print("Done optimizing model")
    
    # Check if GUI is available
    gui_available = True

    return quantized_model, device, gui_available


    

def process_single_image(model, device, frame):
    """
    This image takes a single frame and performs object detection on it and returns the detections with the class, bbox and distance
    """
    
    DOUBLE_SIDE = True
    TRIPLE_SIDE = False
    global OBJECT_DOWNSAMPLE_FACTOR
    down_sample_factor = OBJECT_DOWNSAMPLE_FACTOR
    
    real_widths = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.07,
        'Traffic Light Red': 0.07,
        'Car': 1.7,
        'Person': 0.5,
        'Unknown': 0.5
    }
    real_heights = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.3,
        'Traffic Light Red': 0.3,
        'Car': 1.5,
        'Person': 2.0,
        'Unknown': 0.5
        }
    

    if frame is None:
        print(f"Failed to read the image at {frame}. Skipping.")
        return None

    

    def filter_edges(detection_info, x_value, p_treshold, width):
        max_x = min(x_value + (p_treshold*width),width)
        min_x = max(x_value - (p_treshold*width),0)
        
        for r_dets in detection_info:
            x1, y1, x2, y2 = r_dets['bbox']
            center_x = (x1+x2)/2
            if (center_x < max_x) and (center_x > min_x):
                detection_info.remove(r_dets)
        return detection_info
    
    def apply_offsets(detection_info, offset):
            for det in detection_info:
                x1, y1, x2, y2 = det['bbox']
                x1 = x1 * down_sample_factor
                y1 = y1 * down_sample_factor
                x2 = x2 * down_sample_factor
                y2 = y2 * down_sample_factor
                
                x1 += offset[1]
                y1 += offset[0]
                x2 += offset[1]
                y2 += offset[0]
                det['bbox'] = (x1, y1, x2, y2)
            return detection_info
    
    
    # Generate dictionary with detection details
    detection_info = []
    
    right_detection_info = []
    left_detection_info = []
    middle_detection_info = []
    
    # Crop the top right portion of the image
    top_right_frame = frame[:height // 2, width // 2:]
    
    
    # Ensure the cropped image is resized to the desired dimensions (multiples of model's stride)
    stride = 32
    original_height, original_width = top_right_frame.shape[:2]
    desired_height = (((original_height // stride) + 1) * stride)//down_sample_factor
    desired_width = (((original_width // stride) + 1) * stride) //down_sample_factor
    resized_top_right_frame = cv2.resize(top_right_frame, (desired_width, desired_height))

    # Process the resized cropped frame with YOLOv5 detection
    pred, img_shape = run_inference(model, resized_top_right_frame, device)
    detections = process_detections(pred, resized_top_right_frame, img_shape)
    
    
    
    
    
    # top right detections
    
    
    top_right_offset = (0, width //2)
    
    detections = apply_offsets(detections, top_right_offset)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        #Compute the real widths and heights
        
        real_width_m = real_widths.get(det['class'], 0.5)
        real_height_m = real_heights.get(det['class'], 0.5)
        distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
        right_detection_info.append({
            'class': det['class'],
            'bbox': det['bbox'],
            'distance': distance
        })
        
    if DOUBLE_SIDE or TRIPLE_SIDE:
        
        # Top left detections
        
        # Crop the top left portion of the image
        top_left_frame = frame[:height // 2, :width // 2]
        #Resize
        resized_top_left_frame = cv2.resize(top_left_frame, (desired_width, desired_height))
        #Predict
        pred, img_shape = run_inference(model, resized_top_left_frame, device)
        #Detect
        detections = process_detections(pred, resized_top_left_frame, img_shape)
        
        #Offset
        top_left_offset = (0, 0)
        detections = apply_offsets(detections, top_left_offset)
        
        #Calculate Distance
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            real_width_m = real_widths.get(det['class'], 0.5)
            real_height_m = real_heights.get(det['class'], 0.5)
            distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
            #Store info
            left_detection_info.append({
                'class': det['class'],
                'bbox': det['bbox'],
                'distance': distance
            })
        
        #If I stop the PERSON or CAR class in the left or right image process the middle image
        if 'Person' in [det['class'] for det in left_detection_info] or 'Person' in [det['class'] for det in right_detection_info]:
            TRIPLE_SIDE = True
        elif 'Car' in [det['class'] for det in left_detection_info] or 'Car' in [det['class'] for det in right_detection_info]:
            TRIPLE_SIDE = True
        else: 
            TRIPLE_SIDE = False
                
        if TRIPLE_SIDE:
            #Crop the top middle portion of the image
            top_middle_frame = frame[:height // 2, width // 4:3*width // 4]
            #Resize
            resized_top_middle_frame = cv2.resize(top_middle_frame, (desired_width, desired_height))
            #Predict
            pred, img_shape = run_inference(model, resized_top_middle_frame, device)
            #Detect
            detections = process_detections(pred, resized_top_middle_frame, img_shape)
            
            #Offset
            top_middle_offset = (0, width // 4)
            detections = apply_offsets(detections, top_middle_offset)
            
            #Calculate Distance
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                real_width_m = real_widths.get(det['class'], 0.5)
                real_height_m = real_heights.get(det['class'], 0.5)
                distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
                middle_detection_info.append({
                    'class': det['class'],
                    'bbox': det['bbox'],
                    'distance': distance
                })
                
            #Smooth out the 3 different detections (if right left and middle)
            # Remove any cut off detections and let the middle detection take over it
            right_detection_info = filter_edges(right_detection_info, width//2, 0.15, width)
            left_detection_info = filter_edges(left_detection_info, width//2, 0.15, width)
            
            #Smooth out edges of the middle detection
            #Filter out edges of middle part of the image
            middle_detection_info = filter_edges(middle_detection_info, (width*3//10), 0.05, width)
            middle_detection_info = filter_edges(middle_detection_info, (width*7//10), 0.05, width)
                
    detection_info = left_detection_info + right_detection_info + middle_detection_info
                
    return detection_info




def traffic_object_detection(frame_queue, state_queue, model, device, stop_event):
    """
    This is the main thread of object detection, it initializes the memory then it has a frame
    queue and a state queue, it constantly takes frames from the frames queue and then appends resulting states
    in the states queue
    """
    prev_frame_time = 0

    # Set distance threshold for red light/traffic sign detection
    red_light_distance_threshold = 4.5  # meters
    speed_sign_distance_threshold = 10  # meters
    person_distance_threshold = 10 # meters
    car_distance_threshold = 10 # meters

    # FPS CAP
    global FPS_CAP
    # Memory buffers for red lights and speed signs
    red_light_memory = collections.deque(maxlen=5)
    speed_sign_memory = collections.deque(maxlen=5)
    person_memory = collections.deque(maxlen=10)
    car_memory = collections.deque(maxlen=5)
    
    saw_red_light = False
    last_speed_limit = 10 #  this sets the initial speed of the car
    
    initial_person_position = "None"
    current_person_position = "None"
    
    # Car
    car_spotted = False
    
     # Data Logger
       # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time", 
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]
    
    
        # Create a new folder named with the current date and time
    now = datetime.now()

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)
    folder_name = now.strftime("%m_%d_%H")
    folder_path = os.path.join(log_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    
    # Create a new CSV file within the new folder, also named with the current date and time
    file_name = now.strftime("%Y_%m_%d_%H_%M_%S") + "_detection_log.csv"
    file_path = os.path.join(folder_path, file_name)

    # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time", 
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]
    
    # Create the CSV file and write the headers
    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    print("Data logger Object detection ready")       

    while True:
        if frame_queue:
            # Get the newest frame from the queue
            frame = frame_queue.pop()
            if frame is not None:
                # Record the start time for processing
                start_time = time.time()

                # Process the frame and create a new state
                dets = process_single_image(model, device, frame)

                # Visualize detections
                if dets:
                    for det in dets:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class']} ({det['distance']:.1f}m)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # FPS overlay
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
                prev_frame_time = new_frame_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Show frame
                cv2.imshow("Live Detection Feed", frame)
                cv2.waitKey(1)

                # Record the end time for processing and calculate the duration
                end_time = time.time()
                detect_processing_time = end_time - start_time
    
                # Local loop variables for checking closest traffic lights and signs
                # Speed signs
                bool_saw_speed_sign = False
                closest_seen_speed_limit = 0
                closest_speed_sign_distance = np.inf
    
                # Traffic lights
                closest_traffic_light_distance = np.inf
                bool_saw_red_light = False
                
                #People
                closest_person_distance = np.inf
                
                temp_initial_person_position = "None"
                temp_current_person_position = "None"
                
                # Cars
                temp_car_spotted = False
                closest_car_distance = np.inf
                #TODO: Implement logic to detect cars
                if dets:
                    for det in dets:
                        # First check for traffic lights
                        if det['class'] == 'Traffic Light Red':
                            closest_traffic_light_distance = min(closest_traffic_light_distance, det['distance'])
                            bool_saw_red_light = True
                            
                        elif det['class'] in ['Speed-limit-10km-h', 'Speed-limit-15km-h', 'Speed-limit-20km-h']:
                            if det['distance'] < closest_speed_sign_distance:
                                closest_speed_sign_distance = det['distance']
                                closest_seen_speed_limit = int(det['class'].split('-')[2].replace('km', ''))
                                bool_saw_speed_sign = True
                                
                        elif det['class'] == 'Person':
                            x_center = (det['bbox'][0] + det['bbox'][2])/2
                            middle_bound_left = 0.40
                            middle_bound_right = 0.60
                            closest_person_distance = min(closest_person_distance, det['distance'])
                            #Initialize person position
                            if initial_person_position == "None":
                                temp_initial_person_position = "Right" if x_center > width // 2 else "Left"
                                temp_current_person_position = "Right" if x_center > width // 2 else "Left"
                                
                            #Update person position if it was right
                            elif initial_person_position == "Right":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center < width * middle_bound_left:
                                    temp_current_person_position = "Left"
                                else:
                                    temp_current_person_position = "Right"
                            
                            #Update person position if it was left
                            elif initial_person_position == "Left":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center > width * middle_bound_right:
                                    temp_current_person_position = "Right"
                                else:
                                    temp_current_person_position = "Left"
                        elif det['class'] =='Car':
                            closest_car_distance = min(closest_car_distance, det['distance'])
                            temp_car_spotted = True
                        
                        
    
                # Update memory buffers
                red_light_memory.append(bool_saw_red_light and closest_traffic_light_distance < red_light_distance_threshold)
                speed_sign_memory.append((bool_saw_speed_sign, closest_seen_speed_limit) if closest_speed_sign_distance < speed_sign_distance_threshold else (False, 0))
                person_memory.append((closest_person_distance < person_distance_threshold, temp_current_person_position) if closest_person_distance < person_distance_threshold else (False, "None"))
                car_memory.append(temp_car_spotted and closest_car_distance < car_distance_threshold)
                
                # Check the memory buffers
                saw_red_light = all(red_light_memory)
                # Update last speed limit if all detected speed signs agree
                if all(sign[0] for sign in speed_sign_memory) and len(set(sign[1] for sign in speed_sign_memory)) == 1:
                    last_speed_limit = closest_seen_speed_limit
                
               # Update person positions if all detections agree
                if all(person[0] for person in person_memory):
                    positions = [person[1] for person in person_memory]
                    if len(set(positions)) == 1:
                        current_person_position = positions[0]
                        if initial_person_position == "None":
                            initial_person_position = positions[0]
                
                # Check if person memory is all false
                if not any(person[0] for person in person_memory):
                    initial_person_position = "None"
                    current_person_position = "None"
                    
                # Memory for car spotting
                car_spotted = all(car_memory)
                
                new_state = {
                    "spotted_red_light": saw_red_light,
                    "Speed limit": last_speed_limit,
                    "Initial Person Position": initial_person_position,
                    "Current Person Position": current_person_position,
                    "Car Spotted": car_spotted
                }
                
    
                state_queue.append(new_state)
                #print(f"Updated shared_state: {new_state}")
                end_time = time.time()
                
                traffic_detect_processing_time = end_time - start_time
                
                # Measure CPU usage
                cpu_usage = psutil.cpu_percent(interval=None)
             
                # Extract the current timestamp and break it into components
                now = datetime.now()
                year, month, day = now.year, now.month, now.day
                hour, minute, second = now.hour, now.minute, now.second
                
                
                # Prepare the data for
                detection_info = {
                    "detections": dets,
                    "detect_processing_time": detect_processing_time,
                    "traffic_processing_time": traffic_detect_processing_time,
                    "year": year,
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "second": second,
                    "state": new_state,
                    "cpu_usage": cpu_usage
                }
                
                with open(file_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        detection_info["detections"],
                        detection_info["detect_processing_time"],
                        detection_info["traffic_processing_time"],
                        detection_info["year"],
                        detection_info["month"],
                        detection_info["day"],
                        detection_info["hour"],
                        detection_info["minute"],
                        detection_info["second"],
                        detection_info["state"],
                        detection_info["cpu_usage"]
                    ])
                # If object detection is running too fast, cap the FPS
                if traffic_detect_processing_time < 1 / OBJECT_FPS_CAP:
                    time.sleep(1 / OBJECT_FPS_CAP - traffic_detect_processing_time)
    print("Object detection stopped...(got killed)")

def adjust_throttle(state_queue, throttle_queue, max_car_speed=20):
    """
    This thread reads off the states and decides the throttle for the self driving car
    """
    already_found_car = False
    kill_object_detection = False
    while True:
        if state_queue:
            # Get the newest state from the queue
            new_state = state_queue.pop()
            # Adjust the throttle based on the shared state
            throttle_speed = calculate_throttle_based_on_state(new_state, max_car_speed)
            # Update the throttle queue with the most recent throttle speed
            car_in_range = False
            if new_state['Car Spotted']:
                car_in_range = True
                already_found_car = True
            if (new_state["Speed limit"] == 15) and already_found_car:
                kill_object_detection = True
            if len(throttle_queue) >= 1:
                throttle_queue.pop()
            throttle_state = {
                "throttle": throttle_speed,
                "car in range": car_in_range,
                "kill object detection": kill_object_detection
                }
            
            throttle_queue.append(throttle_state)

            # Prepare the data to be dumped into JSON
            throttle_info = {
                "speed": new_state["Speed limit"],
                "throttle": throttle_speed,
                "saw_red_light": new_state["spotted_red_light"],
                "timestamp": datetime.now().isoformat()
            }

            # Append the data to a JSON log file
            with open(os.path.join("logs", "throttle_log.json"), "a") as f:
                f.write(json.dumps(throttle_info) + "\n")

        time.sleep(0.03)

def calculate_throttle_based_on_state(state,max_car_speed=20):
    """
    This function takes the state as input and calculates the throttle value based on the state
    """
    # Dummy function to calculate throttle speed based on state
    # Replace with your actual throttle calculation logic
    if state["spotted_red_light"]:
        return 0  # Stop if red light is spotted
    
    elif state["Car Spotted"]:
         #TODO: Implement logic to slow down if a car is spotted
         return 10
    
    #elif (state["Current Person Position"] == "Right" or state["Current Person Position"] =="Middle") and state["Initial Person Position"] == "Right":
    #    return 0  # Stop if person is on the right side or on the road and started on the right side
    
    #elif(state["Current Person Position"] == "Left" or state["Current Person Position"] =="Middle") and state["Initial Person Position"] == "Left":
     #   return 0 # Stop if person is on the left side or on the road and started on the left side
    
    else:
        car_speed_km_h = min(state["Speed limit"], max_car_speed)
        if car_speed_km_h == 15:
          car_speed_km_h = 10
          
        return int(car_speed_km_h/max_car_speed *100)  # Throttle speed is calculated as a percentage of max speed of the car(NOT SIGN)

def main():
    """
    Main loop op the self driving car, populates the frame queue and writes to self driving car its steering and
    throttle, it also initializes everything that is needed
    """
    bus = initialize_can()
    
    cameras = initialize_cameras()
    front_camera = cameras["front"]
    
    print('Creating folders...', file=sys.stderr)

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)

    recording_folder_name = "recording " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    recording_folder = os.path.join(log_root, recording_folder_name)
    os.makedirs(recording_folder, exist_ok=True)
    for subdir in cameras.keys():
        os.makedirs(os.path.join(recording_folder, subdir), exist_ok=True)


    can_listener = CanListener(bus)
    can_listener.start_listening()
    image_queue = Queue()
    image_worker = ImageWorker(image_queue, recording_folder)
    ImageWorker(image_queue, recording_folder).start()
    ImageWorker(image_queue, recording_folder).start()
    image_worker.start()
    can_worker = CanWorker(Queue(), recording_folder)
    can_worker.start()

    print('Recording...', file=sys.stderr)
    frames: Dict[str, cv2.Mat] = dict()
    
    #object detection init
    weights_path = 'src/control_modes/autonomous_mode/old_twente_code/v5_model.pt'  # Adjust as necessary
    output_directory_base = 'logs/detection_frames'
    model, device, gui_available = initialize(weights_path, output_directory_base)
    print("GUI available: ", gui_available)
    
    # Shared state and queues
    global shared_state
    shared_state = {
        "spotted_red_light": False,
        "Speed limit": 10,
        "Initial Person Position": "Right", # This can be "Left" or "Right" or "Middle" or "None"
        "Current Person Position": "Right", # This can be "Left" or "Right" or "Middle" or "None"
        "Car Spotted": False
    }
    
    MAX_CAR_SPEED = 20

    # Deques for state and frame queues with a maximum length
    queue_maxsize = 3
    state_queue = deque(maxlen=queue_maxsize)
    frame_queue = deque(maxlen=queue_maxsize)

    # Shared queue for throttle speed
    throttle_queue = deque(maxlen=1)

    # Create a stop event
    object_stop_event = threading.Event()
    
    # Initialize the threads for frame processing and throttle adjustment
    frame_processing_thread = threading.Thread(target=traffic_object_detection, args=(frame_queue, state_queue,model,device,object_stop_event))
    throttle_adjustment_thread = threading.Thread(target=adjust_throttle, args=(state_queue, throttle_queue,MAX_CAR_SPEED))

    # Start the threads
    frame_processing_thread.start()
    throttle_adjustment_thread.start()
    print("Object detection threads started...")

    try:
        # Define messages
        brake_msg = can.Message(arbitration_id=0x110, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)
        steering_msg = can.Message(arbitration_id=0x220, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
        throttle_msg = can.Message(arbitration_id=0x330, is_extended_id=False, data=[0, 0, 0, 0, 0, 0, 0, 0])
        throttle_task = bus.send_periodic(throttle_msg, CAN_MSG_SENDING_SPEED)
        
        sleep(2)
        
        # Start running
        start_time = time.time()
        frame_count = 0
      
        ret, frame = front_camera.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame! Substituting black frame...")
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        hx, hy = getHorizon(frame)
        print("horizon found at",hy)
        countL = 0
        countR = 0
        countMax = 3
        
        #overtaking initialization 
        car_passed = False
        t0 = 0
        tprev = 0
        distance_driven = 0
        turnsens = 0.2 #higher values means sharper turns
        throttle_index = 0
        car_spotted = False
        try:
            while (True):
                #recording part
                ok_count = 0
                values = can_listener.get_new_values()
                timestamp = time.time()
                for side, camera in cameras.items():
                    ok, frames[side] = camera.retrieve()
                    ok_count += ok
                if ok_count == len(cameras):
                    for side, frame in frames.items():
                        image_worker.put((timestamp, side, frame))
                    can_worker.put((timestamp, values))
                for camera in cameras.values():
                    camera.grab()
                
                
                
                #Get camera data
                _, frame = front_camera.read()
                # add frame to frame queue (in cv2 format)
                frame_queue.append(frame)

                # Give it a value of 1 if the throttle queue is empty
                if len(throttle_queue) != 0:
                  throttle_state = throttle_queue.pop()
                  throttle_index = throttle_state['throttle']
                  car_spotted = throttle_state['car in range']
                  
                  kill_object_detection = throttle_state['kill object detection']
                #if kill_object_detection and car_passed:
                #    object_stop_event.set()
                
                throttle_msg.data = [throttle_index, 0, 1, 0, 0, 0, 0, 0]
                #print(detection_info)
                #TODO: Car passing CODE HERE use the car_spotted variable to see if car is within 10 meters of range
                
                
                
                #Steering part
                frame = cv2.resize(frame, (int(width*scale), int(height*scale))) #resize frame if necessary
                lines = getLines(frame) #get initial lines from front frame
                if lines is not None:
                    lines = newLines(lines) #improve lines by clustering lines which are close together and have a similar orientation
                    llines, rlines = splitLines(lines) #select the best left and right line.

                    """
                    if not llines and not rlines:#rlines is not None and llines is not None:
                        print("no llines and rlines")
                        
                        countR = countMax
                        rlines = rlinesOld
                        #wr = max(wr-0.02,0)
                        mr += 10

                        countL = countMax
                        llines = llinesOld
                        #wl = max(wl-0.02,0)
                        ml += 10

                    elif not rlines: #only left line detected
                        if countL == 0:
                            lline = longestLine(llines)
                            x1l, y1l, x2l, y2l = lline
                            llinesOld = llines
                            
                            wl = 1
                            ml = 0
                        else:
                            countL -= 1
                            llines = llinesOld
                            wl = max(wl-turnsens,0)
                            ml += 10
                        wr = max(wr-turnsens,0)
                        mr += 10
                        rlines = rlinesOld
                        countR = countMax
                    elif not llines: #only right line detected
                        if countR == 0:
                            rline = longestLine(rlines)
                            x1r, y1r, x2r, y2r = rline
                            rlinesOld = rlines
                            wr = 1
                            mr = 0
                        else:
                            countR -= 1
                            rlines = rlinesOld
                            wr = max(wr-turnsens,0)
                            mr += 10
                        wl = max(wl-turnsens,0)
                        ml += 10
                        llines = llinesOld
                        countL = countMax
                        #oldlinecountR = 0
                        
                    else:
                        if countL == 0:
                            lline = longestLine(llines)
                            x1l, y1l, x2l, y2l = lline
                            llinesOld = llines
                            wl = 1
                            ml = 0
                        else:
                            countL -= 1
                            llines = llinesOld
                            wl = max(wl-turnsens,0)
                            ml += 10

                        if countR == 0: 
                            rline = longestLine(rlines)
                            x1r, y1r, x2r, y2r = rline
                            rlinesOld = rlines
                            wr = 1
                            mr = 0
                        else:
                            countR -= 1
                            rlines = rlinesOld
                            wr = max(wr-turnsens,0)
                            mr += 10

                    """
                    wl = 1
                    wr = 1
                    Steer_bias = 300

                    #overtaking manoeuvre
                    if car_spotted == True and car_passed == False: 
                        print(".")
                        if t0 == 0:
                            t0 = timestamp
                            tprev = timestamp
                            print("CAR SPOTTED")
                        tnow = timestamp - t0
                        speed = str(struct.unpack(">H", bytearray(values["speed_sensor"][:2]))[0]) if values["speed_sensor"] else ""
                        speed = int(int(speed)/36) #speed to m/s
                        distance_change = speed *(tnow - tprev)
                        distance_driven += distance_change
                        
                        tprev = tnow
                        if distance_driven <5 : #first turn left
                            target = findTarget(llines, rlines, hy, frame, wl, wr,weight = 0, bias = -400, draw = 0)
                            print("Going Left")
                        elif distance_driven <12: # stay straight for a while
                            target = findTarget(llines, rlines, hy, frame, wl, wr,weight = 1, bias = 0, draw = 0)
                            print("Going Straight")
                        elif distance_driven <17: #turn right to previous lane
                            target = findTarget(llines, rlines, hy, frame, wl, wr,weight = 0, bias = 400 , draw = 0)
                            print("Going Right")
                        else:
                            print("Overtaking Completed")
                            car_passed = True
                    else:
                        target = findTarget(llines, rlines, hy, frame, wl, wr,weight = 1, bias = 0, draw = 0)#, ml = ml, mr = mr)#,bias = -212, draw = 1)
                    
                    
                    #target = findTarget(llines, rlines, hy, frame, wl, wr,weight = 1, bias = 0, draw = 0)#, ml = ml, mr = mr)#,bias = -212, draw = 1)


                    if target == False:
                        print("ERROR, NO LINES FOUND 1")
                        throttle_msg.data = [1, 0, 1, 0, 0, 0, 0, 0]
                        #steering_msg = [0]*8
                        steer_angle = 0
                    else:
                        Error = target - width/2
                        if Error > 0:
                            steer_angle = min(Error/(width/2),1.05)
                        else:
                            steer_angle = max(Error/(width/2),-1.05)
            
                        #ja
    
                        #print("error:", Error)
                else:
                    #if no lines are found slowly drive forward
                    print("ERROR, NO LINES FOUND 2")
                    throttle_msg.data = [1, 0, 1, 0, 0, 0, 0, 0]
                    #steering_msg = [0]*8
                    steer_angle = 0
                #if steer_angle % 10 == 0:
                #print("steering angle", steer_angle)
                steering_msg.data = list(bytearray(struct.pack("f", float(steer_angle)))) + [0]*4
                steering_task.modify_data(steering_msg)
                throttle_task.modify_data(throttle_msg)
                
                #activate the breaks if throttle speed is 0
                if throttle_msg.data[0] == 0:
                  brake_msg.data = [50, 0, 1, 0, 0, 0, 0, 0]
                  brake_task.modify_data(brake_msg)
                else:
                  #Reset the breaks if throttle speed isn't 0
                  brake_msg.data = [0, 0, 1, 0, 0, 0, 0, 0]
                  brake_task.modify_data(brake_msg)
                #cv2.imshow('Camera preview', frame)
                #cv2.waitKey(1)
                
                
                #frame = np.resize(frame[:,:,::-1]/255, (1, 144, 256, 3)).astype(np.float32)
                
                #steering_angle, throttle, brake = predict(session, frame)

                #brake_msg.data = [int(99*max(0, brake))] + 7*[0]
                #steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle)))) + [0]*4
                #throttle_msg.data = [int(99*max(0, throttle)), 0, 1] + 5*[0]

                #brake_msg.modify_data(brake_msg)
                #steering_task.modify_data(steering_msg)
                #throttle_task.modify_data(throttle_msg)
                
                frame_count += 1
        except KeyboardInterrupt:
            pass

        end_time = time.time()
        time_diff = end_time - start_time

        print(f'Time elapsed: {time_diff:.2f}s')
        print(f'Frames processed: {frame_count}')
        print(f'FPS: {frame_count/time_diff:.2f}')
        print('Stopping...', file=sys.stderr)
        can_listener.stop_listening()
        for camera in cameras.values():
            camera.release()
        image_worker.stop()
        can_worker.stop()


    finally:
        throttle_task.stop()
        steering_task.stop()
        brake_task.stop()



if __name__ == '__main__':
    #Detection modus for the object detection part
    main()
    ##session = rt.InferenceSession('drive.onnx')
    ##main(session)

