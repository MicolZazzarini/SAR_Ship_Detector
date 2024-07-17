from yolov10.ultralytics import YOLOv10
import cv2

model_path = 'weights/yolov10s_200e_32b_SGD_best.pt'
model = YOLOv10(model_path)
results = model("img/CAESAR/nave.jpg")
cv2.imshow('img.jpg', results)


# yolo task=detect mode=predict model="weights/yolov10s_200e_32b_SGD_best.pt" source="img/CAESAR/nave.jpg" show=True imgsz=640 name=inference show_labels=True
