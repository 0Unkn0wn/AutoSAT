import cv2
from detection import ObjectDetection
from traffic import TrafficManager

def main(weights_path: str, input_source: str, video_path: str = None):
    object_detector = ObjectDetection(weights_path, input_source)
    traffic_manager = TrafficManager()
    
    if input_source == 'video' and video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Default to webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = object_detector.detect_objects(frame)
        traffic_state = traffic_manager.process_traffic_signals(detections)
        print("Traffic State:", traffic_state)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_label = det['class']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_label} ({det['distance']:.2f}m)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Traffic Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('v5_model.pt', 'video', 'trash.mp4')
