import tensorflow as tf
import numpy as np
import cv2
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# โหลดโมเดล SSD ที่ฝึกไว้
PATH_TO_MODEL = r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity06-2\models\ssd_mobilenet_v2\exported_model\saved_model"
detection_model = tf.saved_model.load(PATH_TO_MODEL)

# กำหนดค่า DeepSORT
tracker = DeepSort(max_age=30, n_init=3, nn_budget=70)

def detect_objects(frame, model):
    """ ตรวจจับวัตถุในภาพ """
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detection_classes = detections['detection_classes'].astype(np.int64)
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']

    return detection_boxes, detection_classes, detection_scores

# กำหนดประเภทของเหรียญ
ALLOWED_CLASSES = {
    1: "10 Bath",
    2: "5 Bath",
    3: "2 Bath",
    4: "1 Bath"
}

# โหลดวิดีโอ
cap = cv2.VideoCapture(r"C:\Users\sirap\OneDrive\Desktop\672-Lab\activity07\coins1.MOV")
width = cap.get(3)
height = cap.get(4)

FONT_SCALE = 2e-3
THICKNESS_SCALE = 1e-3
font_scale = min(width, height) * FONT_SCALE
thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

# ใช้เก็บ track_id ที่เคยนับไปแล้ว
counted_objects = set()
coin_count = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลง BGR เป็น RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ตรวจจับเหรียญ
    boxes, classes, scores = detect_objects(rgb_frame, detection_model)

    detections = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] in ALLOWED_CLASSES:
            box = boxes[i]
            y1, x1, y2, x2 = (box * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])).astype(int)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, scores[i], classes[i]))

    # อัปเดตตัวติดตาม
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        obj_class = track.det_class
        bbox = track.to_tlbr()  # แปลงเป็น (x1, y1, x2, y2)

        if obj_class in ALLOWED_CLASSES:
            x1, y1, x2, y2 = map(int, bbox)  # แปลงค่าเป็น int
            label = f"{ALLOWED_CLASSES[obj_class]}"

            # วาดกรอบสี่เหลี่ยมรอบเหรียญ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # แสดงชื่อเหรียญ
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 0), thickness)

            # นับเหรียญที่ยังไม่ถูกนับ
            if track_id not in counted_objects:
                counted_objects.add(track_id)
                coin_count[ALLOWED_CLASSES[obj_class]] += 1

    # แสดงจำนวนเหรียญแต่ละประเภท
    y_offset = 40
    for i, (coin, count) in enumerate(coin_count.items()):
        y_offset += 40
        cv2.putText(frame, f"{coin}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), thickness)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Coin Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
