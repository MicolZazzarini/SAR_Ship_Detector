from ultralytics import YOLOv10
from PIL import Image
import cv2
save_name_img: str = "prova"
model_path = '../weights/yolov10s_200e_32b_SGD_best.pt'
model = YOLOv10(model_path)
results = model("../img/CAESAR/nave.jpg")

for result in results:
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_img)
    result_img.save(f"{save_name_img}_inference.jpg")
