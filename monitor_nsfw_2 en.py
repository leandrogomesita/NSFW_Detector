import os
import sys
import time
import cv2
import numpy as np
import mss
import winsound
import threading
import tensorflow as tf
import tensorflow_hub as hub
import win32gui
import win32con

# Settings
MODEL_FILE = r"C:\\Users\\leand\\OneDrive\\Desktop\\Matthew\\NSFW_Detector\\model_2\\mobilenet_v2_140_224.1\\mobilenet_v2_140_224\\saved_model.h5"
NSFW_THRESHOLD = 0.4  # Detection threshold (lower = more sensitive)
CAPTURE_INTERVAL = 0.5  # Interval between captures in seconds
WINDOW_NAME = "NSFW Info Monitor"
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 300

# Model categories
CATEGORIES = ["drawings", "hentai", "neutral", "porn", "sexy"]

# Function to load the NSFW model
def load_model():
    print(f"Loading NSFW model from {MODEL_FILE}...")
    try:
        # Register custom TF Hub layer
        custom_objects = {'KerasLayer': hub.KerasLayer}
        model = tf.keras.models.load_model(MODEL_FILE, custom_objects=custom_objects)
        print("Model successfully loaded!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Function to perform NSFW prediction
def predict_nsfw(model, img):
    try:
        # Preprocess the image
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Process results
        scores = dict(zip(CATEGORIES, predictions[0]))
        nsfw_score = scores.get('porn', 0) + scores.get('sexy', 0) + scores.get('hentai', 0)
        top_category = max(scores.items(), key=lambda x: x[1])[0]
        is_nsfw = nsfw_score > NSFW_THRESHOLD
        
        return is_nsfw, nsfw_score, scores, top_category
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False, 0.0, dict(zip(CATEGORIES, [0, 0, 1, 0, 0])), "neutral"

# Function to alert when adult content is detected
def play_warning_sound():
    winsound.Beep(1200, 200)
    time.sleep(0.1)
    winsound.Beep(800, 300)

# Function to create the information window
def create_info_window(is_nsfw, nsfw_score, scores, top_category):
    # Create a blank image to display information
    info_img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255  # White background
    
    # Set colors
    color = (0, 0, 255) if is_nsfw else (0, 150, 0)  # Red if NSFW, green if safe
    bg_color = (240, 240, 240)  # Light gray for background
    
    # Fill background
    info_img[:] = bg_color
    
    # Add top colored bar
    cv2.rectangle(info_img, (0, 0), (WINDOW_WIDTH, 50), color, -1)
    
    # Add main status
    status_text = "ADULT CONTENT DETECTED!" if is_nsfw else "Safe content"
    cv2.putText(info_img, status_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Add NSFW score
    cv2.putText(info_img, f"NSFW score: {nsfw_score:.4f} (Threshold: {NSFW_THRESHOLD})", 
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add top category
    cv2.putText(info_img, f"Top category: {top_category}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Show detailed scores
    y_pos = 150
    cv2.putText(info_img, "Scores per category:", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Show each category with progress bar
    for i, (category, score) in enumerate(scores.items()):
        y_pos = 180 + i * 25
        
        # Category name
        cv2.putText(info_img, f"{category}:", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Progress bar
        bar_length = int(score * 300)
        bar_color = (0, 0, 255) if category in ['porn', 'sexy', 'hentai'] and score > 0.15 else (0, 200, 0)
        cv2.rectangle(info_img, (120, y_pos-15), (120 + bar_length, y_pos-5), bar_color, -1)
        
        # Numeric value
        cv2.putText(info_img, f"{score:.4f}", 
                   (430, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return info_img

# Function to get the information window position
def get_info_window_position():
    try:
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        if hwnd:
            rect = win32gui.GetWindowRect(hwnd)
            x, y, w, h = rect
            return (x, y, w - x, h - y)
        return None
    except Exception as e:
        print(f"Error getting window position: {e}")
        return None

# Main function to monitor the screen
def monitor_screen():
    # Load model
    model = load_model()
    
    # Set up screen capture
    with mss.mss() as sct:
        # Set up information window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Position the window at top-right corner
        screen_width = sct.monitors[0]["width"]
        info_window_x = screen_width - WINDOW_WIDTH - 20
        info_window_y = 50
        cv2.moveWindow(WINDOW_NAME, info_window_x, info_window_y)
        
        # Avoid repeated sound alerts
        last_alert_time = 0
        alert_cooldown = 3  # seconds between alerts
        
        print("Monitoring started! Press 'q' to exit.")
        
        # For the first frame
        is_nsfw, nsfw_score, scores, top_category = False, 0.0, dict(zip(CATEGORIES, [0, 0, 1, 0, 0])), "neutral"
        
        while True:
            try:
                # Get current information window position
                info_window_pos = get_info_window_position()
                
                # Capture the full screen
                screenshot = sct.grab(sct.monitors[0])
                img = np.array(screenshot)
                img = img[:, :, :3]  # Convert BGRA to BGR
                
                # Resize to improve performance
                height, width = img.shape[:2]
                scale_factor = 0.5
                resized = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
                
                # If info window is visible, avoid analyzing that area
                if info_window_pos:
                    x, y, w, h = info_window_pos
                    # Scale window coordinates to resized image
                    x_scaled = int(x * scale_factor)
                    y_scaled = int(y * scale_factor)
                    w_scaled = int(w * scale_factor)
                    h_scaled = int(h * scale_factor)
                    
                    # Check if coordinates are within image bounds
                    if (x_scaled >= 0 and y_scaled >= 0 and 
                        x_scaled + w_scaled <= resized.shape[1] and 
                        y_scaled + h_scaled <= resized.shape[0]):
                        
                        # Replace the window area with a black rectangle
                        resized[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = 0
                
                # Predict NSFW content
                is_nsfw, nsfw_score, scores, top_category = predict_nsfw(model, resized)
                
                # Create information window
                info_window = create_info_window(is_nsfw, nsfw_score, scores, top_category)
                
                # Display information window
                cv2.imshow(WINDOW_NAME, info_window)
                
                # Alert if adult content is detected (with cooldown)
                if is_nsfw:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        threading.Thread(target=play_warning_sound).start()
                        last_alert_time = current_time
                
                # Exit on 'q' key press
                if cv2.waitKey(int(CAPTURE_INTERVAL * 1000)) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(0.5)
    
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    print("Starting NSFW info monitor...")
    print(f"Detection threshold: {NSFW_THRESHOLD} (lower = more sensitive)")
    print("Press 'q' to exit")
    monitor_screen()
