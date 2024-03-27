import torch
import math
from ultralytics import YOLO
import cv2
import time

start_time = time.time()  # 録画開始時刻を記録
record_duration = 120  # 30秒

cap = cv2.VideoCapture(1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter('output.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("weights/best.pt")
classNames = ["Wood scraps", "Soft plastic", "Hard plastic", "Paper", "Cardboard", "PVC", "Metal-based", "shoukyaku", "FRP", "GW", "can", "shuredder", "zappin", "zassen", "sekkouA", "touki"]

while True:
    success, img = cap.read()
    if not success:
        break  # 画像の読み込みに失敗したらループ
    results = model(img, stream=True)
    if time.time() - start_time > record_duration:
        break  # 30秒経過したらループを終了

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        out.write(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
out.release()
cap.release()  # カメラリソースを解放
cv2.destroyAllWindows()


