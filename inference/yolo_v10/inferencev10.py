from ultralytics import YOLOv10
import os
from PIL import Image
import cv2
save_name_img: str = "prova"
model_path = '../weights/yolov10s_200e_32b_SGD_best.pt'
model = YOLOv10(model_path)
results = model("../img/CAESAR/nave.jpg")
# cv2.imwrite('img.jpg', results)
# Save inference results
for result in results:
    result_img = result.plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_img)
    # result_path = os.path.join(results_dir, f"{save_name_img}_inference.jpg")
    result_img.save(f"{save_name_img}_inference.jpg")


# yolo task=detect mode=predict model="weights/yolov10s_200e_32b_SGD_best.pt" source="img/CAESAR/nave.jpg" show=True imgsz=640 name=inference show_labels=True