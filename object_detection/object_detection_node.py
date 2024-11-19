import cv2
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import warnings 

# Supress FutureWarning message
warnings.filterwarnings('ignore', category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Publisher to control turtle velocity
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Set target object for detection
        self.target_object = 'cell phone'  # Replace with desired object label
        self.timer = self.create_timer(0.1, self.detect_and_publish)  # Timer to repeatedly call detect_and_publish
        
        # Open the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open the webcam.")
    
    def detect_objects(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cords

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image from webcam.")
            return
        
        # Perform object detection
        results = self.detect_objects(frame)
        labels, cords = results
        
        # Check if the target object is detected
        detected = False
        distance = None
        for i in range(len(labels)):
            if model.names[int(labels[i])] == self.target_object:
                detected = True
                # Calculate distance based on bounding box size
                row = cords[i]
                x1, y1, x2, y2, conf = row
                width = x2 - x1 
                height = y2 - y1 
                size = width * height
                # Estimate distance to object
                distance = 1 / (size + 1e-6) # Add very small value to prevent zero div
                self.get_logger().info(f"Estimated distance to object {self.target_object}: {distance:.2f}")
                break
        
        # Publish a velocity command based on detection
        twist = Twist()
        if detected:
            twist.linear.x = 1.0  # Move forward if the target object is detected
            self.get_logger().info(f"{self.target_object} detected. Moving turtle forward.")
        else:
            twist.linear.x = 0.0  # Stop if the object is not detected
            #self.get_logger().info(f"{self.target_object} not detected. Turtle is stopped.")
        
        self.publisher_.publish(twist)
        
        # Display the detection result on the webcam feed
        frame = self.plot_boxes(results, frame)
        cv2.imshow('YOLOv5 Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

    def plot_boxes(self, results, frame):
        labels, cords = results
        for i in range(len(labels)):
            row = cords[i]
            x1, y1, x2, y2, conf = row
            if conf >= 0.3:
                h, w, _ = frame.shape
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[int(labels[i])]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
