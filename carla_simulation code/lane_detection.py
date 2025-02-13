import torch
import cv2
import numpy as np
import carla
import time
from ultralytics import YOLO
from vehicle_setup import VehicleSetup  # Import vehicle setup

# Load YOLO Model for Lane Detection with Segmentation
class LaneDetector:
    def __init__(self, model_path="models/yolov8n-seg.pt"):  # Use YOLOv8-Seg for segmentation
        self.model = YOLO(model_path)
        print("[INFO] YOLO Segmentation Model loaded for lane detection.")

    def detect_lanes(self, image):
        """Detect lanes using YOLO segmentation and return lane masks."""
        image_resized = cv2.resize(image, (640, 640))  # Resize for YOLO
        results = self.model(image_resized)

        # Extract segmentation masks
        lane_masks = []
        for result in results:
            for mask in result.masks.xy:  # Extract segmented lane areas
                mask = np.array(mask, dtype=np.int32)
                lane_masks.append(mask)

        return lane_masks

def lane_centering_control(lane_masks, image_width):
    """Computes steering adjustment based on lane position using segmentation masks."""
    if not lane_masks:
        return 0  

    lane_centers = [np.mean(mask[:, 0]) for mask in lane_masks if len(mask) > 0]  # Extract x-coordinates
    if len(lane_centers) == 0:
        return 0

    avg_lane_center = sum(lane_centers) / len(lane_centers)
    center_offset = avg_lane_center - (image_width / 2)
    steering_adjustment = -center_offset / (image_width / 2)  
    return steering_adjustment

def process_image(image, lane_detector):
    """Processes image for lane segmentation & lane centering."""
    image.convert(carla.ColorConverter.Raw)
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img_array.reshape((image.height, image.width, 4))[:, :, :3]

    # Detect Lane Segments
    lane_masks = lane_detector.detect_lanes(img)
    steering_adjustment = lane_centering_control(lane_masks, image.shape[1])
    print(f"[INFO] Steering Adjustment: {steering_adjustment:.2f}")

    # Overlay lane masks on the original image
    for mask in lane_masks:
        cv2.fillPoly(img, [mask], (0, 255, 0))  # Green color for lane segmentation

    cv2.imshow("Segmented Lane Detection", img)
    cv2.waitKey(1)

if __name__ == "__main__":
    setup = VehicleSetup()

    # Destroy existing vehicles before spawning a new one
    setup.destroy()

    setup.spawn_vehicle()
    setup.attach_sensors()
    
    # Allow camera to initialize
    time.sleep(2)  

    lane_detector = LaneDetector()
    front_camera = setup.get_sensor("front_camera")

    if front_camera is None:
        print("[ERROR] No camera found! Exiting...")
        setup.destroy()
        exit()

    front_camera.listen(lambda image: process_image(image, lane_detector))

    try:
        while True:
            pass  
    except KeyboardInterrupt:
        pass
    finally:
        setup.destroy()
