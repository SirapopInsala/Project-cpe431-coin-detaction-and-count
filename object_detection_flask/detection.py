import tensorflow as tf
import numpy as np
import cv2
import os
import collections
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# ตรวจสอบว่ามี Model และ Label Map หรือไม่
SAVED_MODEL_PATH = r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity06-2\models\ssd_mobilenet_v2\exported_model\saved_model"
LABEL_MAP_PATH = r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity06-2\models\ssd_mobilenet_v2\train\label_map.pbtxt"

if not os.path.exists(SAVED_MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {SAVED_MODEL_PATH}")

if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"❌ Label map not found at: {LABEL_MAP_PATH}")

# โหลดโมเดล
print("⏳ Loading the model...")
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)
print("✅ Model loaded successfully!")

# กำหนดมูลค่าของเหรียญแต่ละประเภท (ต้องตรงกับ label_map.pbtxt)
COIN_VALUES = {
    1: 10,   # เหรียญ 10 บาท
    2: 5,    # เหรียญ 5 บาท
    3: 2,    # เหรียญ 2 บาท
    4: 1     # เหรียญ 1 บาท
}

def detect_objects(image_path, filename, result_folder):
    print(f"📂 Detecting objects in: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found at: {image_path}")

    # โหลดและแปลงภาพเป็น RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # เตรียมข้อมูลภาพสำหรับโมเดล
    input_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # ดึงข้อมูลผลลัพธ์
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    # นับจำนวนเหรียญแต่ละประเภท
    coin_counts = {coin: 0 for coin in COIN_VALUES.keys()}
    total_value = 0.0

    for i in range(num_detections):
        class_id = detections["detection_classes"][i]
        score = detections["detection_scores"][i]

        if score > 0.5 and class_id in COIN_VALUES:
            coin_counts[class_id] += 1
            total_value += COIN_VALUES[class_id]

    # วาดกรอบรอบวัตถุที่ตรวจพบ
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

    # สร้างโฟลเดอร์เก็บผลลัพธ์ถ้ายังไม่มี
    os.makedirs(result_folder, exist_ok=True)

    # บันทึกภาพที่ตรวจจับแล้ว
    result_path = os.path.join(result_folder, filename)
    cv2.imwrite(result_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Detection completed! Results saved at: {result_path}")
    print(f"💰 Coin counts: {coin_counts}")
    print(f"💵 Total Value: {total_value:.2f} บาท")

    return result_path, coin_counts, total_value


