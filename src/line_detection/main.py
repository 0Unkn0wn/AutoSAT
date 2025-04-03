import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from itertools import combinations

from src.line_detection.utils import getColorMask, getRoiMask, filterContours, filterWhite
from src.line_detection.line_processor import clusterLines, combineLines, getLines

class LineFollowingNavigation:
    def __init__(self, width=848, height=480, scale=1):
        self.width = width
        self.height = height
        self.scale = scale

    def newLines(self, lines):
        nlines = []
        if lines is not None:
            clusters = clusterLines(lines, int(self.scale * 10), 15)
            for cluster in clusters:
                newline = combineLines(cluster)
                nlines.append(newline)
            return nlines
        return 0

    def splitLines(self, lines):
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

    def longestLine(self, lines):
        longest = 0
        longestline = None
        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((abs(x2 - x1)) ** 2 + (abs(y2 - y1)) ** 2)
            if length > longest:
                longest = length
                longestline = line
        return longestline


    

    def findTarget(self, llines, rlines, horizonh, img, wl=1, wr=1, weight=1, bias=0, draw=1):
        drawimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if draw == 1 else None
        
        if not llines and not rlines:
            return False
            
        elif not rlines:
            lline = self.longestLine(llines)
            x1l, y1l, x2l, y2l = lline
             
            lineparL = np.polyfit((x1l, x2l), (y1l, y2l), 1)
            horizonxL = round((horizonh - lineparL[1]) / lineparL[0])
            
            if draw == 1:
                cv2.line(drawimg, (x1l, y1l), (x2l, y2l), (50, 200, 200), 3) 
                cv2.circle(drawimg, (horizonxL, horizonh), 1, (50, 200, 200), 3)
            
            target = horizonxL

        elif not llines:
            rline = self.longestLine(rlines)
            x1r, y1r, x2r, y2r = rline
            
            lineparR = np.polyfit((x1r, x2r), (y1r, y2r), 1)
            horizonxR = round((horizonh - lineparR[1]) / lineparR[0])
            
            if draw == 1:
                cv2.line(drawimg, (x1r, y1r), (x2r, y2r), (100, 200, 200), 3)
                cv2.circle(drawimg, (horizonxR, horizonh), 1, (100, 200, 200), 3)
            
            target = horizonxR
        else:
            lline = self.longestLine(llines)
            rline = self.longestLine(rlines)

            x1r, y1r, x2r, y2r = rline
            x1l, y1l, x2l, y2l = lline
        
            lineparR = np.polyfit((x1r, x2r), (y1r, y2r), 1)
            horizonxR = round((horizonh - lineparR[1]) / lineparR[0])
            
            lineparL = np.polyfit((x1l, x2l), (y1l, y2l), 1)
            horizonxL = round((horizonh - lineparL[1]) / lineparL[0])

            # Calculate heights at intersections
            heightL = lineparL[1]
            heightR = lineparR[0] * self.width + lineparR[1]
            
            # Calculate intersection point
            x_h = (lineparR[1] - lineparL[1]) / (lineparL[0] - lineparR[0])
            y_h = x_h * lineparL[0] + lineparL[1]

            # Ensure weights are positive
            wl = max(wl, 0.01)
            wr = max(wr, 0.01)
            
            # Calculate target position
            target = ((horizonxL + horizonxR) / 2) + (heightL - heightR) * weight + bias

            if draw == 1:
                cv2.line(drawimg, (x1r, y1r), (x2r, y2r), (100, 200, 200), 3)
                cv2.line(drawimg, (x1l, y1l), (x2l, y2l), (50, 200, 200), 3)  
                cv2.circle(drawimg, (round(x_h), round(y_h)), 1, (210, 200, 200), 3) 
                cv2.circle(drawimg, (horizonxR, horizonh), 1, (100, 200, 200), 3)  
                cv2.circle(drawimg, (horizonxL, horizonh), 1, (50, 200, 200), 3)
                cv2.circle(drawimg, (int(target), horizonh), 1, (180, 200, 200), 3)

        # Draw center reference point
        # Draw center reference point
        if draw == 1:
            cv2.circle(drawimg, (int(self.width / 2), horizonh), 1, (0, 0, 255), 3)
            drawimg = cv2.cvtColor(drawimg, cv2.COLOR_HSV2BGR)
            return target, drawimg
        
        return target, None
        
    def processFrame(self, img, horizonh=280, weight=1, bias=0, draw=1):
        """Process a frame to find the line following target.
        
        Parameters:
        img (ndarray): Input image frame
        horizonh (int): Height of the horizon line for target calculation
        weight (float): Weight factor for line calculation
        bias (float): Bias adjustment for target position
        draw (int): Flag to enable/disable drawing on the output image
        
        Returns:
        tuple: (target position, visualization image if draw=1)
        """
        # Resize image if needed
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))
            
        # Detect lines in the image
        lines = getLines(img, self.scale, self.height, self.width)
        
        # Process detected lines
        if lines is not None:
            # Get the new processed lines
            nlines = self.newLines(lines)
            
            # Split into left and right lines
            if nlines:
                llines, rlines = self.splitLines(nlines)
                
                # Find target based on detected lines
                return self.findTarget(llines, rlines, horizonh, img, weight=weight, bias=bias, draw=draw)
        
        # No valid lines detected
        if draw == 1:
            return None, img
        return None, None
    
    def calculateSteeringAngle(self, target, mid_point=None):
        """Calculate steering angle based on target position.
        
        Parameters:
        target: Target x-position
        mid_point: Middle point of the frame (if None, use self.width/2)
        
        Returns:
        float: Steering angle in degrees (-30 to 30)
        """
        if target is None:
            return 0.0
            
        if mid_point is None:
            mid_point = self.width / 2
            
        # Calculate offset from center
        offset = target - mid_point
        
        # Convert to steering angle (scale to range -30 to 30 degrees)
        max_angle = 30.0
        steering_angle = (offset / (self.width / 2)) * max_angle
        
        # Limit to max angle
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        
        return steering_angle
    
    def calculateSpeed(self, steering_angle, base_speed=100):
        """Calculate speed
        
        Parameters:
        steering_angle: Current steering angle in degrees
        base_speed: Base speed value
        
        Returns:
        float: Adjusted speed value
        """
        # Reduce speed in turns
        angle_factor = 1.0 - (abs(steering_angle) / 30.0) * 0.5
        speed = base_speed * angle_factor
        
        return max(speed, base_speed * 0.5)
    
    def run(self, img, base_speed=100, draw=1):
        """Run the line following algorithm on a frame.
        
        Parameters:
        img: Input image frame from camera/video
        base_speed: Base speed value
        draw: Flag to enable/disable visualization
        
        Returns:
        tuple: (steering_angle, speed, visualization image if draw=1)
        """
        # Process the frame to find target
        target, viz_img = self.processFrame(img, draw=draw)
        
        # Calculate steering angle
        steering_angle = self.calculateSteeringAngle(target)
        
        # Calculate speed
        speed = self.calculateSpeed(steering_angle, base_speed)
        
        if draw == 1:
            # Add steering and speed info to visualization
            if viz_img is not None:
                text = f"Steering: {steering_angle:.1f}° | Speed: {speed:.1f}"
                cv2.putText(viz_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                center_x = int(self.width / 2)
                center_y = self.height - 50
                endpoint_x = center_x + int(steering_angle * 2)
                cv2.line(viz_img, (center_x, center_y), (endpoint_x, center_y - 30), (0, 255, 0), 3)
                
            return steering_angle, speed, viz_img, endpoint_x
        
        return steering_angle, speed, None, None, None



    def process(self, frame, base_speed=100, draw=1):
        """Process a single frame and return the results."""
        steering_angle, speed, viz_img, end_x = self.run(frame, base_speed, draw)
        return steering_angle, speed, viz_img, end_x

if __name__ == "__main__":
    nav = LineFollowingNavigation(width=848, height=480)
    cap = cv2.VideoCapture("trash.mp4")  # Replace with your actual video file
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        steering_angle, speed, viz_img, end_x = nav.run(frame)
        
        cv2.imshow("Line Following", viz_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()