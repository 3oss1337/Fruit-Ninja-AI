from ultralytics import YOLO
import keyboard
import threading
import bettercam
from pynput.mouse import Button, Controller
import math
import numpy as np
import time
import cv2
import torch 
import pygetwindow as gw
import pyautogui

from pathlib import Path

def load_model(device):
    model = YOLO("C:/Users/Ahmed/Desktop/Fruit Ninja AI/best.pt")
    model.to(device)
    # Print out the class names to verify
    print("Model Class Names:")
    for i, name in enumerate(model.names):
        print(f"{i}: {name}")
    
    # Warm up the model with a dummy input
    dummy_data = torch.zeros((1, 3, 640, 640), device=device)
    model(dummy_data)
    return model


mouse = Controller()

camera = bettercam.create(output_color="BGR", device_idx=0, region=(0,0,1920,1200))

def init_bomb_cords(x1, x2, y1, y2):
    bomb_cords = np.array([x1, y1, x2, y2])
    return bomb_cords

def is_fruit_collided_with_bomb(fruit_box, bomb_list):
    f_x1, f_x2, f_y1, f_y2 = fruit_box
    for bomb in bomb_list:
        b_x1, b_x2, b_y1, b_y2 = bomb
        if (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y1 <= b_y2) or \
           (b_x1 <= f_x1 <= b_x2 and b_y1 <= f_y2 <= b_y2) or \
           (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y1 <= b_y2) or \
           (b_x1 <= f_x2 <= b_x2 and b_y1 <= f_y2 <= b_y2):
            return True
    return False    

def set_safe_fruits(fruits, bombs):
    safe_fruits = []
    print(f"\nProcessing {len(fruits)} detected fruits")
    for fruit in fruits:
        center_x, center_y, width, height = fruit

        # Debug print for each fruit
        print(f"Fruit: center_x={center_x}, center_y={center_y}, width={width}, height={height}")

        if center_y > 1000:
            print(f"Skipping fruit below y=1000")
            continue
        
        fruit_x1 = center_x - width / 2
        fruit_y1 = center_y - height / 2
        fruit_x2 = center_x + width / 2
        fruit_y2 = center_y + height / 2
        fruit_box = (fruit_x1, fruit_y1, fruit_x2, fruit_y2)
        
        if not is_fruit_collided_with_bomb(fruit_box, bombs):
            safe_fruits.append((center_x, center_y))
            print(f"Added safe fruit at ({center_x}, {center_y})")
        else:
            print(f"Fruit collides with bomb, not safe")
    
    print(f"Total safe fruits: {len(safe_fruits)}")
    return safe_fruits

def sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass

cached_cos_sin = {}

def move_mouse(radius, num_steps):
    if radius not in cached_cos_sin:
        angles = np.linspace(0, 2 * np.pi, num_steps)
        cos_vals = np.cos(angles) * radius
        sin_vals = np.sin(angles) * radius
        cached_cos_sin[radius] = (cos_vals, sin_vals)
    else:
        cos_vals, sin_vals = cached_cos_sin[radius]

    x_positions = mouse.position[0] + cos_vals
    y_positions = mouse.position[1] + sin_vals

    # move the mouse in a circle
    for i in range(len(cos_vals)):
        new_x = x_positions[i]
        new_y = y_positions[i]
        mouse.position = (new_x, new_y)
        sleep(0.000001)  

def run_bot(safe_fruits):
    if not safe_fruits:
        print("No safe fruits found.")
        return

    for fruit_location in safe_fruits:
        fruit_x, fruit_y = fruit_location
        radius = 50  
        num_steps = 50 

        # Debug log for fruit location
        print(f"Moving to fruit at: {fruit_x}, {fruit_y}")
        mouse.position = (fruit_x, fruit_y)  
        mouse.press(Button.left)  
        move_mouse(radius, num_steps)  
        mouse.release(Button.left)  
    print("Bot finished slicing fruits.")

# def visualize_detections(frame, results):
#     """
#     Draw bounding boxes and labels on the frame for visualization
#     """
#     if results[0].boxes is not None:
#         for box, cls in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist()):
#             x1, y1, x2, y2 = map(int, box)
#             label = results[0].names[int(cls)]
#             conf = results[0].boxes.conf.tolist()[0]
            
#             # Choose color based on class
#             color = (0, 255, 0) if label == 'fruit' else (0, 0, 255)
            
#             # Draw rectangle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
#             # Put label and confidence
#             label_text = f'{label}: {conf:.2f}'
#             cv2.putText(frame, label_text, (x1, y1-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#     return frame

def take_screenshot(stop_event, model, device, display_window=True):
    """
    Captures and performs detection on screenshots with optional visualization
    # """
    # cv2.namedWindow("Model Detection View", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Model Detection View", 1280, 720)

    while not stop_event.is_set():
        # Capture screenshot
        screenshot = camera.grab(region=(0, 0, 1920, 1200))  # Capture screenshot
        
        if screenshot is None:
            continue

        # Perform detection
        results = model(source=screenshot, 
                       device=device, 
                       verbose=True,  # Enable verbose output for more info
                       iou=0.25,    # Adjust IOU threshold
                       conf=0.3,    # Lowered confidence threshold for more detections
                       imgsz=(640,640))

        detected_fruits, detected_bombs = [], []

        # Parse detection results
        for box, cls in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist()):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Debug print for each detection
            print(f"Detection: {results[0].names[int(cls)]} at ({center_x}, {center_y})")

            # Separate detected objects into fruits and bombs
            if results[0].names[int(cls)] == "bomb":
                detected_bombs.append(init_bomb_cords(x1, y1, x2, y2))
            elif results[0].names[int(cls)] == "fruit":
                detected_fruits.append((center_x, center_y, width, height))

        # Visualize detections if display_window is True
        # if display_window:
        #     frame_with_detections = visualize_detections(screenshot.copy(), results)
        #     cv2.imshow("Model Detection View", frame_with_detections)

        # Run the bot on safe fruits
        safe_fruits = set_safe_fruits(detected_fruits, detected_bombs)
        run_bot(safe_fruits)  # Perform slicing action on safe fruits

        # Print detection stats
        print(f"Detected Fruits: {len(detected_fruits)}, Detected Bombs: {len(detected_bombs)}")

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(device)  # load the detection model
    stop_event = threading.Event()
    
    screenshot_thread = threading.Thread(target=take_screenshot, 
                                         args=(stop_event, model, device))
    screenshot_thread.start()
    
    keyboard.wait("q")  # Wait for 'q' key to stop the bot
    stop_event.set()
    screenshot_thread.join()
    camera.release()

if __name__ == "__main__":
    main()