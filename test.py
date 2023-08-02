from ultralytics import YOLO
import cv2

model = YOLO("yolov8_detect_gb.pt")

# model.predict(source='GBG01.jpg', show=True, save=True) 
# model.predict(source='GBR01.jpg', show=True, save=True) 
# model.predict(source='GBY01.jpg', show=True, save=True) 
# model.predict(source='LBB01.jpg', show=True, save=True) 
# model.predict(source='LBW01.jpg', show=True, save=True) 
# model.predict(source='LBY01.jpg', show=True, save=True) 


# model.predict(source='Rotation_Image_Test.jpg', show=True, save=True)
import time 


url = "http://admin:AIruca88@192.168.1.47:80/axis-cgi/mjpg/video.cgi?"

cap = cv2.VideoCapture(url)
# print(cap.get(cv2.CAP_PROP_FPS)) #25
# print(cap.set(cv2.CAP_PROP_FPS, 10))
print(cap.get(cv2.CAP_PROP_FPS))
while(cap.isOpened()):
    
    ret, frame = cap.read()
    frame = frame[588:2100, 0:1512]
    frame = cv2.resize(frame, (640 , 640))
    # sh = frame.shape #2688, 1512
    # print("Size:", sh)
    model.predict(source=frame, show=True, save=True)
    # cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    # time.sleep(0.02)



cap.release()
cv2.destroyAllWindows()
