import cv2
import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes, check_img_size
from yolov5.utils.torch_utils import select_device

class ObjectDetection:
    def __init__(self, weights_path: str, input_source: str = 'video'):
        self.device = select_device('cpu')
        self.model = DetectMultiBackend(weights_path, device=self.device, fp16=False)
        self.model.warmup()
        self.input_source = input_source  # 'video' or 'camera'
        self.real_dims = {
            'Traffic Light Red': (0.07, 0.3),
            'Traffic Light Green': (0.07, 0.3),
            'Speed-limit-10km-h': (0.6, 0.6),
            'Speed-limit-15km-h': (0.6, 0.6),
            'Speed-limit-20km-h': (0.6, 0.6),
            'Car': (1.7, 1.5),
            'Person': (0.5, 1.9),
            'Object': (0.5, 0.5),
        }
        
        self.labels = {
            0: 'Car',
            1: 'Person',
            2: 'Speed-limit-10km-h',
            3: 'Speed-limit-15km-h',
            4: 'Speed-limit-20km-h',
            5: 'Traffic Light Green',
            6: 'Traffic Light Red'
        }

    def estimate_distance(self, x1, y1, x2, y2, class_label):
        focal_length = 540
        real_width, real_height = self.real_dims.get(class_label, (0.5, 0.5))
        box_width, box_height = abs(x2 - x1), abs(y2 - y1)
        if box_width <= 0 or box_height <= 0:
            return 0
        distance = (focal_length * (real_width + real_height) / 2) / ((box_width + box_height) / 2)
        return max(distance - 0.3, 0)

    def detect_objects(self, frame):
        img_size = check_img_size(frame.shape[:2], s=32)
        img = cv2.resize(frame, (img_size[1], img_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(np.expand_dims(img, axis=0)).float() / 255.0
        img = img.to(self.device)

        with torch.no_grad():
            pred = self.model(img)
        
        det = non_max_suppression(pred, 0.7, 0.45, max_det=1000)[0]
        detections = []
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_label = self.labels.get(class_id, 'Unknown')
                if class_label in self.real_dims:
                    distance = self.estimate_distance(x1, y1, x2, y2, class_label)
                    detections.append({'class': class_label, 'confidence': conf.item(), 'bbox': [x1, y1, x2, y2], 'distance': distance})
        return detections
