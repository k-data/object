from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))


    model = YOLO("weights/best.pt")
    classNames = ["Wood scraps", "Soft plastic", "Hard plastic", "Paper", "Cardboard", "PVC", "Metal-based",
                  "shoukyaku", "FRP", "GW", "can", "shuredder", "zappin", "zassen", "sekkouA", "touki"]
    while True:
        success, img = cap.read()
        if not success:
            break
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
    cap.release()

path = "video/M2U00004_trimmed.mp4"
for img in video_detection(path):
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'で終了
        break

cv2.destroyAllWindows()
