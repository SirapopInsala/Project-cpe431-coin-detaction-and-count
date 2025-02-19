import tensorflow as tf
import numpy as np
import cv2
import os
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Check if the model and label map exist
SAVED_MODEL_PATH = r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity06-2\models\ssd_mobilenet_v2\exported_model\saved_model"
LABEL_MAP_PATH = r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity06-2\models\ssd_mobilenet_v2\train\label_map.pbtxt"

if not os.path.exists(SAVED_MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {SAVED_MODEL_PATH}")

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"❌ Label map not found at: {LABEL_MAP_PATH}")

# Load the model
print("⏳ Loading the model...")
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)
print("✅ Model loaded successfully!")

# Object detection function
def detect_objects(image_path, filename):
    print(f"📂 Detecting objects in: {image_path}")

    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found at: {image_path}")

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare the input tensor for the model
    input_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # Extract detection results
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    # Draw bounding boxes around detected objects
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_rgb,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=0.5,
        agnostic_mode=False,
    )

    # ✅ Create the results folder if it doesn't exist
    result_folder = "object_detection_flask/static/results"
    os.makedirs(result_folder, exist_ok=True)

    # ✅ Save the detection results
    result_path = os.path.join(result_folder, filename)
    cv2.imwrite(result_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Detection completed! Results saved at: {result_path}")
    return result_path
