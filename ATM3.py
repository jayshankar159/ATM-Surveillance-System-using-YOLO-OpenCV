import cv2
import os
import datetime
import time
import winsound
from ultralytics import YOLO


video_path = r"C:\Users\Anand Shankar\PycharmProjects\PythonProject\YOLO\FILE VIDEO\ATM.mp4"
save_folder = r"C:\Users\Anand Shankar\PycharmProjects\PythonProject\YOLO\snapshots"
os.makedirs(save_folder, exist_ok=True)

motion_cooldown = 5
beep_frequency = 1000
beep_duration = 700

model = YOLO("yolov8n.pt")

last_alert_time = 0
suspicious_objects = ["person", "gun", "knife", "cigarette", "mask", "helmet"]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    results = model(frame)[0]
    alert = False

    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        cls_name = model.names[int(cls_id)]
        if cls_name in suspicious_objects:
            alert = True

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, cls_name.upper(), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    current_time = time.time()
    if alert and (current_time - last_alert_time) > motion_cooldown:

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(save_folder, f"suspicious_{timestamp}.jpg")
        cv2.putText(frame, "SUSPICIOUS!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.imwrite(snapshot_path, frame)
        print(f"Alert! Snapshot saved: {snapshot_path}")


        winsound.Beep(beep_frequency, beep_duration)

        last_alert_time = current_time

    cv2.imshow("ATM Surveillance", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
